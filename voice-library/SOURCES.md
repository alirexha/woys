# Voice library — provenance

Audit trail for the starter voice library imported via `scripts/voice_library_import.py`.

Models are NOT bundled in this repo (per `LICENSE` / brief §7). They live in `~/.local/share/woys/models/` on the user's machine. This file documents the upstream HuggingFace URL + slug for each.

| Slug | Display | Status | Upstream URL |
|------|---------|--------|--------------|
| `donald_trump` | Donald Trump (POTUS) | ✅ | <https://huggingface.co/Hazza1/DonaldTrump/resolve/main/Trump.zip> |
| `e_girl` | E-Girl (HQ Female) | ✅ | <https://huggingface.co/ZokaxDesu/e-girl/resolve/main/e-girl.zip> |
| `lana_del_rey` | Lana Del Rey (NFR Era) | ✅ | <https://huggingface.co/pinguG/Lana-Del-Rey/resolve/main/NFR.zip> (brief listed `LanaDelReyV2.zip`; actual file in repo is `NFR.zip` — recovered manually) |
| `harley_quinn` | Harley Quinn V2 (Enemy Within, Titan Pretrain) | ✅ | <https://huggingface.co/Cauthess/HarleyQuinnTitanPretrain/resolve/main/Harley%20Quinn%20Version%202%20-%20Enemy%20Within.zip> |
| `catwoman` | Catwoman (Laura Bailey) | ✅ | <https://huggingface.co/Cauthess/CatwomanLauraBailey/resolve/main/Catwoman%20-%20Laura%20Bailey.zip> |
| `megan_fox` | Megan Fox | ✅ | <https://huggingface.co/dragoncrack/https___www_donationalerts_com_r_crack_dragon/resolve/main/MeganFox.zip> |
| `spongebob_persian` | SpongeBob Persian Dub (Bab Asfanji) | ✅ | <https://huggingface.co/PlushymehereJC/Spongebob_Persian_dub/resolve/main/Bab_Asfanj.zip> |
| `jennie` | Jennie (BLACKPINK, Legacy Core 32K 230E) | ✅ | <https://huggingface.co/natanworkspace/Legacy_Core_Models/resolve/main/JENNIE_Legacy_Core1.5_32K_230E.zip> |

**Removed in v0.6.2** (user opted out):
- `alfred_pennyworth` — was at <https://huggingface.co/Homiebear/AlfredPennyworth_465e_8835s/resolve/main/AlfredPennyworth_465e_8835s.zip>
- `batman_troy_baker` — was at <https://huggingface.co/Zogii/zogiiRVC/resolve/main/Bruce%20Wayne%20(Troy%20Baker)%20Batman%20The%20Telltale%20Series%20(RVC%20v2)%20400%20Epochs.zip>

Models are typically OpenRAIL-licensed or unlicensed; treat as **personal use only**. Do NOT redistribute the weights without checking each upstream repo's specific license.
