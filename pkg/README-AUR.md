# Publishing `vcclient-cachy` to the AUR

The PKGBUILD and .SRCINFO in this directory are submission-ready for the
Arch User Repository. **Publication is gated on the GitHub repo being
publicly accessible** — the AUR uses unauthenticated `git clone`, so a
private source URL won't work.

## Pre-flight

1. **Make the repo public** (or add a public mirror):
   ```
   gh repo edit alirexha/vcclient-cachy --visibility public --accept-visibility-change-consequences
   ```
   *(See `LICENSE` first — root LICENSE is currently "All Rights Reserved";
   re-publishing the repo means anyone can clone it, but they still can't
   redistribute under permissive terms.)*

2. **Have an AUR account** at https://aur.archlinux.org/register/.

3. **Upload your SSH public key** to your AUR account profile.

## Submission

```
# 1. Clone the empty AUR repo for this package
git clone ssh://aur@aur.archlinux.org/vcclient-cachy.git /tmp/aur-vcclient-cachy
cd /tmp/aur-vcclient-cachy

# 2. Copy the pre-built bundle
cp ~/ai/vcclient-cachy/pkg/PKGBUILD .
cp ~/ai/vcclient-cachy/pkg/.SRCINFO .

# 3. Verify on the AUR side
makepkg --printsrcinfo > .SRCINFO   # regenerate just in case

# 4. Stage + commit + push
git add PKGBUILD .SRCINFO
git commit -m "vcclient-cachy 0.2.0: initial AUR upload"
git push origin master
```

After push, the package is live at `https://aur.archlinux.org/packages/vcclient-cachy`.

## Updating

When you cut a new version:

```
# In the main repo
sed -i 's/^pkgver=.*/pkgver=0.3.0/' pkg/PKGBUILD
cd pkg && makepkg --printsrcinfo > .SRCINFO

# In the AUR clone
cp ~/ai/vcclient-cachy/pkg/PKGBUILD .
cp ~/ai/vcclient-cachy/pkg/.SRCINFO .
git commit -am "vcclient-cachy 0.3.0"
git push origin master
```

## Local install test (without publishing)

`makepkg -s` from `pkg/` will fail today because:
- the `source=` line points at a `git+https://github.com/alirexha/vcclient-cachy.git#tag=v0.2.0`
- the repo is private, so the unauthenticated git clone bombs

To smoke-test the PKGBUILD logic locally without the network roundtrip:

```
mkdir -p /tmp/vcclient-cachy-test/vcclient-cachy-0.2.0
cp -a ~/ai/vcclient-cachy/{src,pkg,pyproject.toml,README.md,LICENSE,upstream,docs} \
      /tmp/vcclient-cachy-test/vcclient-cachy-0.2.0/
cp ~/ai/vcclient-cachy/pkg/PKGBUILD /tmp/vcclient-cachy-test/
cd /tmp/vcclient-cachy-test
# Override the source array via env so makepkg uses our local copy:
PKGBUILD_SOURCE_OVERRIDE=local makepkg -s --noconfirm --nodeps
```

(That env-var trick is not standard makepkg behavior; the easiest "test
build" is to wait until the repo is public, or temporarily flip the
`source=` URL to a published mirror.)
