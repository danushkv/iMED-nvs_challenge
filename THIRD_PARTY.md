# Third-party software and licensing

This release combines MIT-licensed original project code with research
software maintained by other projects. The root MIT license does not replace
or override the licenses and attribution requirements of those components.

## Endo-4DGS and Gaussian Splatting

Method 2 is an overlay on the Endo-4DGS repository, whose root currently
contains the Gaussian-Splatting research license. That license restricts use
to research/evaluation and requires the license and attribution notices to be
retained when redistributing derivative work. The assembly script copies that
license into the Method 2 snapshot.

Upstream reference used for the local baseline tree:

```text
repository: https://github.com/smbonilla/Endo-4DGS.git
commit:     d4127d1b8f588a2c0e69a9b6fd24a9fafa2fcced
```

Confirm that this is the intended release base before publishing Method 2.
The sibling Method 2 directory was an unversioned working snapshot, so its
curated overlay—not an invented commit identifier—is what is preserved.

## gsplat and G-SHARP

Method 4 imports the upstream G-SHARP surgical trainer from gsplat:

```text
repository: https://github.com/nerfstudio-project/gsplat.git
commit:     846c07932a77a901b474c40dd7fbfe42965ab354
license:    Apache-2.0
```

Prefer a Git submodule pinned to that commit or clone the exact commit during
environment setup. Do not replace it with an unpinned `main` checkout.

## GeoSCOPE container parents

The final GeoSCOPE Docker context uses two historical Synapse images:

```text
docker.synapse.org/syn74277461/imed-nvs-baseline:v1
docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
```

The first supplies the licensed Endo-4DGS runtime. The second supplies the
exact camera/reprojection helpers used by M3-B. Image layers are not committed
to this repository. The mutable `latest` tag should be replaced by a recorded
digest for archival reproducibility when possible.

## Dataset and challenge material

The iMED challenge dataset, target RGB frames, organizer Docker images, and
challenge evaluation package are not licensed by this repository. They are
not included in the release snapshot. Follow the organizer's access and
redistribution terms.

## Release obligations

1. Retain the full Endo-4DGS/Gaussian-Splatting license with Method 2.
2. Retain gsplat's Apache-2.0 license and notices if vendoring its source.
3. Include citations requested by Endo-4DGS, Gaussian Splatting,
   gsplat/G-SHARP, and the iMED challenge in publications using the code.
4. Preserve the provenance manifests accompanying the permitted
   prediction-only qualitative assets.
5. Apply the root MIT license only to original release code; do not assume it
   automatically covers a third-party component.
