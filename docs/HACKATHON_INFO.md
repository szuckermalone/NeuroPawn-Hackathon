# Hackathon Info: Neurotech @ NC State BCI Hackathon '26

This summarizes the organizer handouts for our team. The original files aren't in the repo (see the note at the bottom).

## Our team

| | |
|---|---|
| **Team** | 14 |
| **Track** | NeuroPawn |
| **Hacking room** | 2236 (NeuroPawn teams 11–15) |
| **Track room** (briefings, judging) | 2207 |

**Members:** Yamin Alshayeb (NCSU, ECE), Zaid Abdeen (NCSU, Physics/ECE), Emily Propst (WCU, Psychology/Cell Bio), Colin Liu (Duke, BME), Sebastian Zucker-Malone (Elon, CSC)

## Submission (due **Sun Nov 8, 11:45 AM**, no exceptions)

You'll need:
- [ ] Team number (14)
- [ ] Team name
- [ ] GitHub repo link. **The repo must be public.**
- [ ] Presentation as a downloaded **.pptx** file (in `presentation/`)

NeuroPawn submission form: https://docs.google.com/forms/d/e/1FAIpQLScpI2IxceXE7n6V3zQoq_8cgLJ5YL1XhKJdXNjJP7JUNEFGXg/viewform?usp=header

## Schedule

**Location:** Engineering Building 3, 1840 Entrepreneur Drive, Raleigh, NC 27606
**Discord:** https://discord.gg/7ufV353jV

### Saturday, Nov 7 (9:00 AM – 10:00 PM)
| Time | What | Where |
|---|---|---|
| 9:00 – 9:45 AM | Check-in, name tags, breakfast (Bruegger's) | Main lobby |
| 9:45 – 10:30 AM | Kick-off ceremony | Main room 2124 |
| 10:30 – 11:00 AM | Sponsor track briefing | NeuroPawn: 2207 |
| 11:00 AM | **Hacking begins** | NeuroPawn: 2236 / 2240 |
| 12:15 – 1:00 PM | Lunch (Little Caesars) | Main room 2124 |
| 1:00 – 10:00 PM | Hacking; track room open | NeuroPawn: 2207 |
| 10:00 PM | Day ends | |

### Sunday, Nov 8 (9:00 AM – 2:30 PM)
| Time | What | Where |
|---|---|---|
| 8:00 – 9:30 AM | Doors open, breakfast (Panera) | Main room 2124 |
| 9:30 – 11:45 AM | Hacking; **submissions due 11:45** | NeuroPawn: 2207 |
| 12:00 – 1:15 PM | Sponsor talks, lunch, judging (teams called to track rooms one at a time) | 2124 / 2207 |
| 1:30 – 2:30 PM | Closing ceremony | Main room 2124 |

## NeuroPawn headset setup

**In the box:** headset frame with chin strap, main board (Knight board), recording electrodes, comfort-padding electrodes (soft black screw-in tips that hold the recording electrodes against your head), ground electrode (ear clip), conductive paste, USB-C cable, and paper instructions.

1. Download the **EXG Visualizer** for your OS from the NeuroPawn website (enter your email to get the download). If your OS warns that it's untrusted, choose "run anyway".
2. Connect the headset to your laptop over USB-C. Lights should turn on on the Arduino and on the board (move the cover to see the board light). The colors can vary.
3. In the visualizer, pick the headset's serial port from the dropdown at the top right. Signals should appear in the top-left panel.
   - The team whose box is labeled **IMU** must select the **Knight Board IMU** board type, which gives an extra data stream.
4. Plug the recording electrodes into the **positive rail** of pins. This is a common mistake, so double-check the pin number and the rail. For each channel you want on, flip the switch next to it *away* from the header.
5. Screw the electrodes into the headset frame.
6. Turn on channels **one at a time**: click **Active** and **Route RLD** on each, and wait for its signal before moving on.
7. Plug the ground electrode into the **RLD** and **COMM** pins and clip it to your ear.
8. You should now have 7–8 channels streaming.

**Signal quality tips:** try different electrode positions and comfort-pad placements to maximize scalp contact. A little conductive paste on each recording electrode helps.

**Getting data into code:** see the NeuroPawn GitHub for libraries for your language. The organizers say this is the hardest step for most teams. (NC State students get Google Gemini for free.)

## Other track (for reference): ANT Neuro

Movement-based EEG apps using ANT's 24-channel eego systems. Live data comes over **LSL** (`pylsl`), and teams book 30-minute rotations at the hardware stations (Sat 1:00–8:30 PM, Sun 9:30–12:00). Not our track.

---
*Source: the "Hackathon Participant Resources" folder from the organizers. It stays local and is gitignored because it contains the full participant roster (personal info) and copyrighted vendor manuals.*
