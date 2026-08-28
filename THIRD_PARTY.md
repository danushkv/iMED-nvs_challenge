# Third-party software and licensing

This release combines original challenge adaptation code with research
software maintained by other projects. Keeping the repository private does
not remove the obligation to preserve their licenses and attribution.

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

## Dataset and challenge material

The iMED challenge dataset, target RGB frames, organizer Docker images, and
challenge evaluation package are not licensed by this repository. They are
not included in the release snapshot. Follow the organizer's access and
redistribution terms.

## Before making the repository public

1. Retain the full Endo-4DGS/Gaussian-Splatting license with Method 2.
2. Retain gsplat's Apache-2.0 license and notices if vendoring its source.
3. Add citations requested by Endo-4DGS, Gaussian Splatting, gsplat/G-SHARP,
   and the iMED challenge.
4. Confirm that qualitative images may be redistributed.
5. Add an explicit license for original code; do not assume a third-party
   license automatically covers it.
