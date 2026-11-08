"""Build the 5-slide NeuroPawn talk as a .pptx (video embedded, speaker notes).

Canvas mirrors the 1920x1080 web deck: 1 px = 1/144 in, font px / 2 = pt.
"""
import html
import re
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

REPO = Path(sys.argv[1])
DECK5 = Path(sys.argv[2])
OUT = Path(sys.argv[3])
FIG = REPO / "presentation/figures/final"
VID = REPO / "presentation/video"

DARK, LIGHT = "14213D", "F6F5F1"
ORANGE, ORANGE_DK, ORANGE_LT = "E0602D", "B8481C", "F08A5D"
BLUE, BLUE_LT, SOFT, BODY, RULE = "2F5DA8", "9DB8E6", "BFC8D6", "4B5565", "34456A"
HEAD, TEXT = "Space Grotesk", "Inter"


def px(v):
    return Inches(v / 144)


def rgb(h):
    return RGBColor.from_string(h)


def notes_for(sid):
    s = (DECK5 / "project/slides" / f"{sid}.html").read_text()
    return html.unescape(re.search(r"<aside>(.*?)</aside>", s, re.S).group(1))


def text(slide, x, y, w, h, runs, size, color, font=TEXT, bold=False,
         spacing=None, line=None, anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str, or list of paragraphs; each paragraph a str or list of (text, overrides)."""
    tb = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if line:
            p.line_spacing = line
        if i:
            p.space_before = Pt(size * 0.6)
        segs = para if isinstance(para, list) else [(para, {})]
        for seg, ov in segs:
            r = p.add_run()
            r.text = seg.replace("‑", "-")
            f = r.font
            f.name = ov.get("font", font)
            f.size = Pt(ov.get("size", size))
            f.bold = ov.get("bold", bold)
            f.color.rgb = rgb(ov.get("color", color))
            if spacing:
                r._r.get_or_add_rPr().set("spc", str(int(spacing * 100)))
    return tb


def picture(slide, path, x, y, w, h, alt, border=None):
    """Place an image 'contain'-fitted and centred in the px box."""
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    fw, fh = iw * s, ih * s
    pic = slide.shapes.add_picture(str(path), px(x + (w - fw) / 2), px(y + (h - fh) / 2),
                                   px(fw), px(fh))
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    if border:
        pic.line.color.rgb = rgb(border)
        pic.line.width = Pt(0.75)
    return pic


def new_slide(prs, bg, sid):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    s.notes_slide.notes_text_frame.text = notes_for(sid)
    return s


def eyebrow(slide, label, color, x=128, y=128, w=1664):
    text(slide, x, y, w, 36, label, 12, color, bold=True, spacing=1.5, name="Eyebrow")


def title(slide, label, color, x=128, y=180, w=1664, h=150, size=32):
    text(slide, x, y, w, h, label, size, color, font=HEAD, bold=True, line=1.05, name="Title")


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
prs.core_properties.title = "NeuroPawn EEG Drowsiness Detection: 3-minute talk"
prs.core_properties.author = "Team 14"

# 1 ── What we built
s = new_slide(prs, LIGHT, "built")
eyebrow(s, "TEAM 14 · NEUROPAWN TRACK · WHAT WE BUILT IN 24 HOURS", ORANGE_DK)
title(s, "A live EEG drowsiness detector and a drive it can interrupt", DARK)
picture(s, FIG / "gui_signal_check.png", 128, 350, 816, 500,
        "NeuroPawn EEG app with eight live EEG traces, connection badges and the Signal Check table",
        border="DAD7CE")
picture(s, FIG / "game_driving.png", 976, 350, 816, 500,
        "Night Drive: first-person night highway with a steering wheel")
text(s, 128, 884, 816, 90, [[("Lossless 8-channel EEG app: ", {"bold": True, "color": DARK}),
                             ("every sample recorded exactly, live signal-quality scores", {})]],
     13, BODY, line=1.15)
text(s, 976, 884, 816, 90, [[("Pre-sleep detector + Night Drive: ", {"bold": True, "color": DARK}),
                             ("theta must replace alpha; an alert interrupts the drive", {})]],
     13, BODY, line=1.15)

# 2 ── Live demo video
s = new_slide(prs, DARK, "video")
eyebrow(s, "LIVE · REAL DATA FROM OUR FINAL RUN", ORANGE_LT, x=96, y=64, w=1728)
title(s, "The headset sees the brain change state, in sync with the drive", LIGHT,
      x=96, y=108, w=1728, h=130, size=28)
poster = VID / "frame_445.png"
vx, vy, vw, vh = 96, 262, 1280, 720
# Poster as a plain picture underneath: Google Slides drops embedded video on import,
# so the slide still shows the frame there; PowerPoint plays the movie on top.
picture(s, poster, vx, vy, vw, vh, "Still frame of the demo video")
mv = s.shapes.add_movie(str(VID / "night_drive_eeg_demo.mp4"), px(vx), px(vy), px(vw), px(vh),
                        poster_frame_image=str(poster), mime_type="video/mp4")
mv.name = "Demo video"
mv._element.nvPicPr.cNvPr.set(
    "descr", "15-second clip from the final run: Night Drive, live theta/alpha and alpha amplitude, "
             "seven scrolling EEG channels; eyes-closed alpha fades at a tone, the car drifts and the "
             "driver corrects in 1.69 s")
beats = [("0–4 s · eyes closed", ORANGE_LT, "Alpha fills P3, P4 and O1; θ/α sits at 0.26"),
         ("4 s · tone, eyes open", ORANGE_LT, "Alpha fades within seconds; θ/α climbs past 1"),
         ("13–15 s · drift", BLUE_LT,
          "Car drifts right; the driver corrects in 1.69 s. Every event is stamped into the EEG file")]
for i, (head, col, body) in enumerate(beats):
    y = 272 + i * 210
    text(s, 1424, y, 400, 44, head, 14, col, font=HEAD, bold=True)
    text(s, 1424, y + 50, 400, 140, body, 12, SOFT, line=1.2)

# 3 ── Alpha
s = new_slide(prs, LIGHT, "alpha")
eyebrow(s, "VALIDATION 1 · REAL BRAIN SIGNAL", ORANGE_DK)
title(s, "Closing the eyes switches on alpha, and we see it", DARK)
picture(s, FIG / "alpha_session3.png", 128, 340, 1060, 596,
        "Pilot session: spectrum with an 11.25 Hz alpha peak with eyes closed; alpha power x10.6 parietal "
        "and x11.1 occipital; alpha amplitude jumps within two seconds of closing the eyes")
stats = [("×10.6", ORANGE_DK, "parietal alpha with eyes closed, pilot session"),
         ("×15", ORANGE_DK, "posterior alpha in the final run's baseline (θ/α 1.76 → 0.12)"),
         ("11.25 Hz", BLUE, "individual alpha peak, 0.75 Hz wide: brain rhythm, not mains noise")]
for i, (num, col, lab) in enumerate(stats):
    y = 356 + i * 196
    text(s, 1244, y, 548, 80, num, 32, col, font=HEAD, bold=True)
    text(s, 1244, y + 84, 548, 90, lab, 12, BODY, line=1.15)

# 4 ── Fatigue
s = new_slide(prs, LIGHT, "fatigue")
eyebrow(s, "REANALYSIS · THE MOST USABLE TREND", ORANGE_DK)
title(s, "Classic fatigue indices rose as reactions slowed", DARK)
picture(s, FIG / "fatigue_index_trend.png", 128, 340, 1664, 540,
        "Over the 20-minute drive, (theta+alpha)/beta and alpha/beta rise 100-200% above the first three "
        "minutes, central beta power falls about 60%, and reaction time rises on a second axis")
text(s, 128, 900, 1664, 110,
     [[("(θ+α)/β and α/β rose 100–200% (time trend ρ ≈ +0.5, p < 0.05), driven by ", {}),
       ("falling beta", {"bold": True, "color": DARK}),
       (", a known fatigue sign. They co-trend with reaction time but don't track it minute to minute "
        "(time-removed ρ ≈ 0).", {})]], 13, BODY, line=1.2)

# 5 ── Sleep-EDF validation
s = new_slide(prs, LIGHT, "sleepedf")
eyebrow(s, "VALIDATION 2 · REAL SLEEP ONSETS, PUBLIC DATA", ORANGE_DK)
title(s, "Tested on 13 real, expert-scored sleep onsets", DARK)
picture(s, FIG / "sleep_edf_validation.png", 128, 330, 866, 610,
        "Sleep-EDF results across 13 recordings aligned to expert-scored sleep onset: theta/alpha dips while "
        "relaxed in bed then rises sharply through onset; the mean detector score rises from zero at onset; "
        "alerts cluster in the first minutes after onset, with three earlier alerts during scored wake")
for i, (num, col, lab) in enumerate([
        ("9 / 13", ORANGE_DK, "sleep onsets caught, median 1.5 min after the expert-scored onset"),
        ("3 in 5.2 h", BLUE, "false alarms while lying awake in bed")]):
    y = 350 + i * 200
    text(s, 1050, y, 742, 80, num, 32, col, font=HEAD, bold=True)
    text(s, 1050, y + 84, 742, 90, lab, 12, BODY, line=1.15)
text(s, 1050, 780, 742, 170,
     [[("Not solved yet: ", {"bold": True, "color": DARK}),
       ("3 false alarms per hour in active daytime wake. PhysioNet Sleep-EDF, 2 channels; same detector "
        "code, θ and α from Pz-Oz.", {})]], 12, BODY, line=1.2)

# 6 ── Takeaways
s = new_slide(prs, DARK, "takeaways")
eyebrow(s, "TAKEAWAYS", ORANGE_LT)
cols = [("×15", ORANGE_LT, "alpha when the eyes close: a low-cost headset reads real brain state"),
        ("0", BLUE_LT, "false alarms in 20 min of driving, with 98% of the EEG usable"),
        ("+59%", ORANGE_LT, "reaction time as the EEG fatigue index rose 100–200%")]
for i, (num, col, lab) in enumerate(cols):
    x = 128 + i * (522 + 49)
    text(s, x, 250, 522, 170, num, 66, col, font=HEAD, bold=True, anchor=MSO_ANCHOR.BOTTOM)
    text(s, x, 440, 522, 130, lab, 15, LIGHT, line=1.2)
ln = s.shapes.add_connector(1, px(128), px(650), px(1792), px(650))
ln.line.color.rgb = rgb(RULE)
ln.line.width = Pt(0.75)
text(s, 128, 700, 1664, 110,
     [[("Limits: ", {"bold": True, "color": LIGHT}),
       ("one participant drove; real sleep onsets come from 13 public 2-channel recordings, and daytime "
        "false alarms (3 per hour) still need work.", {})]], 14, SOFT, line=1.2)
text(s, 128, 820, 1664, 110,
     [[("Next: ", {"bold": True, "color": LIGHT}),
       ("our own headset on sleep-restricted drivers, longer drives, and a webcam PERCLOS reference on the same clock.", {})]],
     14, SOFT, line=1.2)

prs.save(OUT)
print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
