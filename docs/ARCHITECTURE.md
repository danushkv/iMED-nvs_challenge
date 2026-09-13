# GeoSCOPE architecture

GeoSCOPE combines two frozen source-only predictors.

## Geometry branch

For each synchronized Endoscope2/L RGB-D frame, M3-B:

1. backprojects metric depth into the source camera;
2. estimates local surface normals and tangent footprints;
3. transforms surfels into the Endoscope1 target camera using supplied
   calibration;
4. applies depth-aware surface splatting and visibility;
5. propagates valid values by at most three pixels;
6. returns RGB and the exact filled geometric support mask.

## Learned branch

Method 2 trains the challenge-adapted Endo-4DGS model per sequence using only
Endoscope2 RGB, metric depth, source tool masks, calibration, and timestamps.
Its accepted addition is metric-depth L1 supervision with weight `5e-5`.

## Selective completion

At native target resolution:

```python
valid = nearest_resize(m3b_filled_valid_mask)
final = m3b_rgb.copy()
final[~valid] = method2_rgb[~valid]
assert byte_equal(final[valid], m3b_rgb[valid])
```

There is no blending, learned selector, RGB-black hole detector, target mask,
or target-image access. The fallback is selected solely by source-derived
geometric support.
