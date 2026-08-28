# Visual assets

`method_lineage.svg` is a hand-authored overview of the experiment lineage.

The `qualitative/` gallery is populated only on explicit request by:

```bash
bash imed_nvs_release/scripts/collect_qualitative_assets.sh
```

That script copies existing model predictions unchanged. It does not compose,
resize, recolor, or generate imagery, and it never copies Endoscope1 ground
truth. Review challenge dataset and derived-output redistribution terms before
committing the gallery.

For a paper-quality comparison, use the same sequence/frame for every method,
add GT only after obtaining redistribution permission, state whether images are
640×512 evaluation renders or 1280×1024 challenge exports, and include error
maps made by the frozen offline evaluator rather than manually enhanced crops.
