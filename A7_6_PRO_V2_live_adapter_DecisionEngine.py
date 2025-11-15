
from __future__ import annotations

import os, sys, time, json, uuid, math, traceback, signal, random, weakref, logging, threading as _threading
import urllib.request, contextlib, webbrowser, subprocess
# ✅ GEREKLİ IMPORTLAR
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import math
import os
import threading as _threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable, List, Tuple, Sequence
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime, timezone, date
from logging.handlers import RotatingFileHandler
from time import monotonic as _mono  # R1-4: monotonic zaman
from collections import deque  # R2-1: karar/sinyal izi için
import logging

logger = logging.getLogger(__name__)


# ========= Dış Execution/Risk sınıfları (opsiyonel) =========
try:
  from A6_1_PRO_V2_executor_and_metrics import (
    ExecutionConfig as _ExternalExecutionConfig,
    RiskConfig as _ExternalRiskConfig,
  )
except Exception:
  _ExternalExecutionConfig = None
  _ExternalRiskConfig = None

from A7_5_PRO_V2_live_adapter_Classes  import _start_audible_alert, DecisionThresholds, _stop_audible_alert


# ========= Opsiyonel bağımlılıklar (graceful degrade) =========
# Not: Dosyanın başında zaten "import numpy as np" var.
# Buradaki blok, yalnızca "np" tanımsızsa devreye girecek şekilde daraltıldı.
try:
  _ = np  # type: ignore  # np önceden import edildiyse hiçbir şey yapma
except NameError:
  try:
    import numpy as np  # son çare: burada import etmeyi dene
  except Exception:
    class _NPStub:
      @staticmethod
      def clip(x, a, b):
        return max(a, min(b, x))

      @staticmethod
      def isfinite(x):
        return True

      @staticmethod
      def mean(x):
        try:
          return sum(x) / len(x)
        except Exception:
          return 0.0


    np = _NPStub()  # type: ignore

try:
  from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest
except Exception:
  CollectorRegistry = Gauge = Counter = Histogram = generate_latest = None  # type: ignore


# ========= Yardımcılar: güvenli JSON ve bekleme =========

def _num(x, d=None):
  try:
    v = float(x)
    return v if np.isfinite(v) else d
  except Exception:
    return d

import re
from typing import Any

class DecisionEngine:
  def __init__(self, thr: DecisionThresholds, cooldown_bars: int = 0, use_neutral_band: bool = True):
    import math, logging
    log = logging.getLogger("A7_PRO_V2_live_adapter")

    # ---- temel alanlar
    self.thr = thr  # önce atayalım, sonra alanlarını düzenleyeceğiz
    self._policy_entries_blocked = False
    self._policy_degraded = False

    tau = getattr(thr, "tau", None)
    delt = getattr(thr, "delta", None)

    def _apply_safe_policy():
      # Var olan thr_up/thr_down bozuk değilse onlardan türet
      try:
        up = float(getattr(self.thr, "thr_up", 0.5))
        dn = float(getattr(self.thr, "thr_down", 0.5))
        if not math.isfinite(up) or not math.isfinite(dn) or (up <= dn):
          up, dn = 0.5, 0.5  # yalnızca bozuksa nötre çek
        gap = max(0.0, up - dn)

        # min_band: sabit 0.10 YAPMA — gap'ten türet, güvenli aralıkta tut
        mb = getattr(self.thr, "min_band", None)
        if mb is None or not math.isfinite(float(mb)) or float(mb) <= 0.0:
          mb = max(0.02, min(0.20, 0.5 * gap))

        # tau/delta: yoksa türev
        tau = getattr(self.thr, "tau", None)
        delt = getattr(self.thr, "delta", None)
        if tau is None or not math.isfinite(float(tau)):
          tau = dn + 0.5 * gap
        if delt is None or not math.isfinite(float(delt)) or float(delt) <= 0.0:
          delt = gap

        self.thr.thr_up = float(max(0.0, min(1.0, up)))
        self.thr.thr_down = float(max(0.0, min(1.0, dn)))
        self.thr.min_band = float(max(0.0, min(0.49, mb)))
        self.thr.thr_neu = float(max(0.10, min(0.90, float(getattr(self.thr, "thr_neu", 0.40)))))
        self.thr.tau = float(max(0.0, min(1.0, tau)))
        self.thr.delta = float(max(0.0, min(1.0, delt)))
      except Exception:
        # en kötü ihtimal nötr ve makul band
        self.thr.thr_up = 0.5;
        self.thr.thr_down = 0.5
        self.thr.min_band = 0.06;
        self.thr.thr_neu = 0.40
        self.thr.tau = 0.5;
        self.thr.delta = 0.1

    def _clip_and_order():
      # kırp & tutarlılık + realignment görünürlüğü
      old_up = float(self.thr.thr_up)
      old_dn = float(self.thr.thr_down)
      old_band = float(self.thr.min_band)

      self.thr.thr_up = max(0.0, min(1.0, float(self.thr.thr_up)))
      self.thr.thr_down = max(0.0, min(1.0, float(self.thr.thr_down)))
      self.thr.min_band = max(0.0, min(0.49, float(self.thr.min_band)))

      # Order enforcement
      new_dn = min(self.thr.thr_down, self.thr.thr_up - self.thr.min_band)
      changed = (
              (abs(self.thr.thr_down - new_dn) > 1e-12) or
              (abs(old_up - self.thr.thr_up) > 1e-12) or
              (abs(old_dn - self.thr.thr_down) > 1e-12) or
              (abs(old_band - self.thr.min_band) > 1e-12)
      )
      self.thr.thr_down = new_dn

      if changed:
        try:
          self.complog.log("thr_realign", {
            "old": {"up": old_up, "down": old_dn, "band": old_band},
            "new": {"up": float(self.thr.thr_up), "down": float(self.thr.thr_down),
                    "band": float(self.thr.min_band)},
            "reason": "order_enforcement"
          })
        except Exception:
          pass

      if not (self.thr.thr_up > self.thr.thr_down and self.thr.min_band >= 0.0):
        raise ValueError("inconsistent thresholds after alignment")

    if (tau is None) or (delt is None):
      # politika eksik → BLOCK + audible alert
      self._policy_entries_blocked = True
      self._policy_degraded = True
      _apply_safe_policy()
      try:
        log.warning("[A7-THR] policy missing (tau/delta) → entries BLOCKED (safe mode)")
      except Exception:
        print("[A7-THR] policy missing (tau/delta) → entries BLOCKED (safe mode)")
      try:
        _start_audible_alert("[A7-THR] policy missing (tau/delta) → entries BLOCKED (safe mode)")
      except Exception:
        pass
    else:
      # hizalama dene
      try:
        tau = float(tau)
        delt = abs(float(delt))
        if math.isfinite(tau) and math.isfinite(delt) and delt > 0:
          up_p = tau + delt / 2.0
          dn_p = tau - delt / 2.0
          band_p = max(0.02, min(0.49, delt / 2.0))
          self.thr.thr_up = max(float(self.thr.thr_up), float(up_p))
          self.thr.thr_down = min(float(self.thr.thr_down), float(dn_p))
          self.thr.min_band = max(float(self.thr.min_band), float(band_p))
        _clip_and_order()

        # hizalama başarılı → block/alert temizle
        self._policy_entries_blocked = False
        self._policy_degraded = False
        try:
          _stop_audible_alert()
        except Exception:
          pass

      except Exception as e:
        # 1) A6_6'da otomatik re-kalibrasyon dene
        try:
          from A6_6_PRO_V2_live_signal import recalibrate_if_needed as _recal_now
        except Exception:
          _recal_now = None

        tried_recal, recal_ok, recal_info = False, False, {}
        if callable(_recal_now):
          tried_recal = True
          try:
            recal_ok, recal_info = _recal_now(require_platt=True, min_rows=4000)
          except Exception:
            recal_ok = False

        if recal_ok:
          # 2) re-kalibrasyon sonrası tekrar hizalama
          try:
            tu = getattr(self.thr, "tau", None)
            de = getattr(self.thr, "delta", None)
            if tu is None or de is None:
              raise ValueError("tau/delta missing after recalibration")
            tu = float(tu);
            de = abs(float(de))
            up_p = tu + de / 2.0
            dn_p = tu - de / 2.0
            band_p = max(0.02, min(0.49, de / 2.0))
            self.thr.thr_up = max(float(self.thr.thr_up), float(up_p))
            self.thr.thr_down = min(float(self.thr.thr_down), float(dn_p))
            self.thr.min_band = max(float(self.thr.min_band), float(band_p))
            _clip_and_order()
            # başarı: blokları kaldır, alert’i kapat
            self._policy_entries_blocked = False
            self._policy_degraded = False
            try:
              _stop_audible_alert()
            except Exception:
              pass
            try:
              log.warning("[A7-THR] re-calibration succeeded → alignment ok.")
            except Exception:
              pass
          except Exception as _e2:
            # 3) yine başaramadı → fail-safe
            self._policy_entries_blocked = True
            self._policy_degraded = True
            _apply_safe_policy()
            try:
              log.warning("[A7-THR] re-alignment after re-calibration failed → entries BLOCKED (safe mode). reason=%s",
                          _e2)
            except Exception:
              pass
            try:
              _start_audible_alert("[A7-THR] alignment after recalibration failed → BLOCKED")
            except Exception:
              pass
        else:
          # re-kalibrasyon yok/başarısız → fail-safe
          self._policy_entries_blocked = True
          self._policy_degraded = True
          _apply_safe_policy()
          try:
            if tried_recal:
              log.warning(
                "[A7-THR] alignment failed; re-calibration not applied/failed → entries BLOCKED (safe mode). first_reason=%s info=%s",
                e, recal_info)
            else:
              log.warning("[A7-THR] policy alignment failed → entries BLOCKED (safe mode). reason=%s", e)
          except Exception:
            pass
          try:
            _start_audible_alert("[A7-THR] policy alignment failed → entries BLOCKED (safe mode)")
          except Exception:
            pass

    # ---- kalan alanlar
    self.last_decision: Optional[str] = None  # "LONG" | "SHORT" | "FLAT" | "HOLD"
    self._cooldown_left: int = 0
    self._cooldown_bars: int = int(max(0, cooldown_bars))
    self._use_neutral_band = use_neutral_band
    self.cooldown_starts: int = 0
    self.flip_prevented_cnt: int = 0

  def _enter_cooldown(self):
    self._cooldown_left = self._cooldown_bars
    if self._cooldown_left > 0:
      self.cooldown_starts += 1

  def _tick_cooldown(self):
    if self._cooldown_left > 0:
      self._cooldown_left -= 1

  def _normalize_probs(self, probs: Dict[str, float]) -> Tuple[float, float, float]:
    pu = float(probs.get("UP", 0.0))
    pd = float(probs.get("DOWN", 0.0))
    pn = float(probs.get("NEU", max(0.0, 1.0 - (pu + pd))))
    s = pu + pd + pn
    if s > 0:
      pu, pd, pn = pu / s, pd / s, pn / s
    return pu, pd, pn

  def _hysteresis_decision(self, pu: float, pd: float) -> str:
    if self.last_decision == "LONG":
      if pu >= max(self.thr.thr_up, pd + self.thr.min_band):
        return "LONG"
      elif pd >= max(self.thr.thr_down, pu + self.thr.min_band):
        return "FLAT"
      else:
        return "HOLD"
    elif self.last_decision == "SHORT":
      if pd >= max(self.thr.thr_down, pu + self.thr.min_band):
        return "SHORT"
      elif pu >= max(self.thr.thr_up, pd + self.thr.min_band):
        return "FLAT"
      else:
        return "HOLD"
    else:
      if pu >= self.thr.thr_up and pu >= pd + self.thr.min_band:
        return "LONG"
      elif pd >= self.thr.thr_down and pd >= pu + self.thr.min_band:
        return "SHORT"
      else:
        return "HOLD"

  def decide(self, probs: Dict[str, float]) -> str:
    # == FAIL-SAFE: politika hizalaması bozuksa (entries blocked) sadece ÇIKIŞ serbest ==
    if getattr(self, "_policy_entries_blocked", False):
      pu, pd, pn = self._normalize_probs(probs)

      # Pozisyon yoksa yeni giriş YOK → HOLD
      if self.last_decision not in ("LONG", "SHORT"):
        return "HOLD"

      # Pozisyon varken karşı eşik sağlanırsa FLAT'a çık; aksi halde HOLD
      if self.last_decision == "LONG":
        if pd >= max(self.thr.thr_down, pu + self.thr.min_band):
          self.last_decision = "FLAT"
          return "FLAT"
        return "HOLD"

      if self.last_decision == "SHORT":
        if pu >= max(self.thr.thr_up, pd + self.thr.min_band):
          self.last_decision = "FLAT"
          return "FLAT"
        return "HOLD"

      # Beklenmeyen durum — güvenli varsayılan
      return "HOLD"

    # == NORMAL YOL ==
    pu, pd, pn = self._normalize_probs(probs)
    decision = "HOLD"

    # Cooldown sırasında flip engelle
    if self._cooldown_left > 0:
      if self.last_decision in ("LONG", "SHORT"):
        decision = self.last_decision
      else:
        decision = "HOLD"
      self.flip_prevented_cnt += 1
      self._tick_cooldown()
      return decision

    # Nötr bant kullanımı + histerezis
    if self._use_neutral_band:
      weak_up = (pu < self.thr.thr_up)
      weak_dn = (pd < self.thr.thr_down)
      small_gap = (abs(pu - pd) < self.thr.min_band)
      if weak_up and weak_dn and small_gap and (pn >= self.thr.thr_neu):
        decision = "HOLD"
      else:
        decision = self._hysteresis_decision(pu, pd)
    else:
      decision = self._hysteresis_decision(pu, pd)

    # R1-8: cooldown sadece FLAT -> (LONG/SHORT) girişinde başlasın
    if decision in ("LONG", "SHORT") and self.last_decision not in ("LONG", "SHORT") and self._cooldown_bars > 0:
      self._enter_cooldown()

    self.last_decision = decision
    self._tick_cooldown()
    return decision

