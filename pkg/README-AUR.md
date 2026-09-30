# Publishing `woys` to the AUR

## Not working yet

The PKGBUILD and .SRCINFO in this directory are a **draft**. A package
built from them installs but does not run. Do not submit it to the AUR
until these are fixed:

- **No Python dependencies.** `depends=()` lists only python, PipeWire and
  nvidia-utils. `python -m installer` does not resolve dependencies, so
  numpy, torch, onnxruntime-gpu, textual, rich, soxr, huggingface_hub,
  tomli-w, sounddevice, librosa and the rest are missing and every real
  subcommand fails with `ModuleNotFoundError`. Either declare the Arch
  `python-*` packages (and test against their versions, which are far
  newer than the pins in `pyproject.toml`) or build a private venv under
  `/opt/woys` in `package()`.
- **No Python version that works.** `pyproject.toml` needs Python
  `>=3.11,<3.13` (the torch / onnxruntime-gpu pins have no newer wheels).
  The PKGBUILD now says so, which means it cannot be installed on current
  Arch, whose `python` is newer. A `python311`/`python312` based venv is
  the likely way out.
- **No `woys-pw-out`.** `build()` does not run `make -C bin` and
  `package()` does not install `bin/woys-pw-out`. woys plays audio through
  that helper by default (`prefer_native_pw = true`) and the engine
  refuses to start without it. Needs `make -C bin` in `build()`,
  `install -Dm755 bin/woys-pw-out "$pkgdir/usr/bin/woys-pw-out"` in
  `package()`, and `gcc` + `pkgconf` in `makedepends`.
- **No way to fetch the foundation weights.** `scripts/download_weights.py`
  is not in the wheel, but the engine's error messages point at it. It
  needs to ship inside the `woys` package (e.g. as a `woys models`
  subcommand) or be installed under `/usr/share/woys`.
- **Generic top-level module names.** The wheel installs `audio`, `tui`
  and `server` straight into site-packages, which can clash with other
  packages in a system-wide install.

Publication is also gated on the GitHub repo being publicly accessible:
the AUR uses unauthenticated `git clone`, so a private source URL won't
work.

The steps below are for once the package works. `<version>` is the
current `__version__` in `src/woys/__init__.py`; `scripts/release.py`
keeps `pkgver` in PKGBUILD and .SRCINFO in step with it.

## Pre-flight

1. **Make the repo public** (or add a public mirror):
   ```
   gh repo edit alirexha/woys --visibility public --accept-visibility-change-consequences
   ```
   *(See `LICENSE` first — root LICENSE is currently "All Rights Reserved";
   re-publishing the repo means anyone can clone it, but they still can't
   redistribute under permissive terms.)*

2. **Have an AUR account** at https://aur.archlinux.org/register/.

3. **Upload your SSH public key** to your AUR account profile.

## Submission

```
# 1. Clone the empty AUR repo for this package
git clone ssh://aur@aur.archlinux.org/woys.git /tmp/aur-woys
cd /tmp/aur-woys

# 2. Copy the PKGBUILD
cp ~/woys/pkg/PKGBUILD .

# 3. Generate .SRCINFO from it
makepkg --printsrcinfo > .SRCINFO

# 4. Stage + commit + push
git add PKGBUILD .SRCINFO
git commit -m "woys <version>: initial AUR upload"
git push origin master
```

After push, the package is live at `https://aur.archlinux.org/packages/woys`.

## Updating

When you cut a new version:

```
# In the main repo: bump src/woys/__init__.py, then
python scripts/release.py          # patches README, PKGBUILD, .SRCINFO
bash scripts/check_version_drift.sh

# In the AUR clone
cp ~/woys/pkg/PKGBUILD .
makepkg --printsrcinfo > .SRCINFO
git commit -am "woys <version>"
git push origin master
```

## Local install test (without publishing)

`makepkg -s` from `pkg/` fails while the repo is private, because the
`source=` line is an unauthenticated
`git+https://github.com/alirexha/woys.git#tag=v<version>` clone.

To smoke-test the PKGBUILD logic locally, lay out the source directory
by hand and point `source=` at nothing:

```
V=<version>
mkdir -p /tmp/woys-test/src/woys-$V
cp -a ~/woys/{src,bin,pkg,docs,pyproject.toml,README.md,LICENSE,NOTICE} \
      /tmp/woys-test/src/woys-$V/
sed -e 's/^source=.*/source=()/' -e 's/^sha256sums=.*/sha256sums=()/' \
    ~/woys/pkg/PKGBUILD > /tmp/woys-test/PKGBUILD
cd /tmp/woys-test
makepkg -e -s --noconfirm    # -e: build from the existing src/ tree
```

Do not add `--nodeps`: the missing dependencies are exactly what this
test needs to show.
