"""macOS: keep the acquisition GUI running at full speed while it's hidden.

When the driving game covers the screen, macOS "App Nap" throttles hidden apps' timers.
That would delay the drowsiness detector exactly when it matters, and idle system sleep
could stop a long recording. This declares a user-initiated activity
(NSProcessInfo.beginActivityWithOptions:reason:) for the life of the process, which is
what `pmset -g assertions` shows. Uses ctypes only (no pyobjc). Does nothing on other OSes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

# NSActivityUserInitiated = 0x00FFFFFF | NSActivityIdleSystemSleepDisabled (1 << 20)
NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF | (1 << 20)
_token = None  # keep the activity object alive


def prevent_app_nap(reason: str = "Recording EEG and monitoring drowsiness") -> bool:
    global _token
    if sys.platform != "darwin" or _token is not None:
        return _token is not None
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        addr = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        # arm64 requires calling objc_msgSend through a correctly typed (non-variadic) prototype
        msg = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(addr)
        msg_cstr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(addr)
        msg_activity = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_uint64, ctypes.c_void_p)(addr)
        sel = objc.sel_registerName
        info = msg(objc.objc_getClass(b"NSProcessInfo"), sel(b"processInfo"))
        text = msg_cstr(msg(objc.objc_getClass(b"NSString"), sel(b"alloc")), sel(b"initWithUTF8String:"),
                        reason.encode("utf-8"))
        token = msg_activity(info, sel(b"beginActivityWithOptions:reason:"), NS_ACTIVITY_USER_INITIATED, text)
        if not token:
            return False
        _token = msg(token, sel(b"retain"))
        return True
    except Exception:
        return False
