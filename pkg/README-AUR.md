# Publishing `woys` to the AUR

The PKGBUILD and .SRCINFO in this directory are submission-ready for the
Arch User Repository. **Publication is gated on the GitHub repo being
publicly accessible** — the AUR uses unauthenticated `git clone`, so a
private source URL won't work.

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

# 2. Copy the pre-built bundle
cp ~/ai/woys/pkg/PKGBUILD .
cp ~/ai/woys/pkg/.SRCINFO .

# 3. Verify on the AUR side
makepkg --printsrcinfo > .SRCINFO   # regenerate just in case

# 4. Stage + commit + push
git add PKGBUILD .SRCINFO
git commit -m "woys 0.13.3: initial AUR upload"
git push origin master
```

After push, the package is live at `https://aur.archlinux.org/packages/woys`.

## Updating

When you cut a new version:

```
# In the main repo
sed -i 's/^pkgver=.*/pkgver=0.13.3/' pkg/PKGBUILD
cd pkg && makepkg --printsrcinfo > .SRCINFO

# In the AUR clone
cp ~/ai/woys/pkg/PKGBUILD .
cp ~/ai/woys/pkg/.SRCINFO .
git commit -am "woys 0.13.3"
git push origin master
```

## Local install test (without publishing)

`makepkg -s` from `pkg/` will fail today because:
- the `source=` line points at a `git+https://github.com/alirexha/woys.git#tag=v0.13.3`
- the repo is private, so the unauthenticated git clone bombs

To smoke-test the PKGBUILD logic locally without the network roundtrip:

```
mkdir -p /tmp/woys-test/woys-0.13.3
cp -a ~/ai/woys/{src,pkg,pyproject.toml,README.md,LICENSE,upstream,docs} \
      /tmp/woys-test/woys-0.13.3/
cp ~/ai/woys/pkg/PKGBUILD /tmp/woys-test/
cd /tmp/woys-test
# Override the source array via env so makepkg uses our local copy:
PKGBUILD_SOURCE_OVERRIDE=local makepkg -s --noconfirm --nodeps
```

(That env-var trick is not standard makepkg behavior; the easiest "test
build" is to wait until the repo is public, or temporarily flip the
`source=` URL to a published mirror.)
