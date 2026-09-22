# Seed for VTOS: the psv2_001 grounded_sam2 baseline pipeline.
# Test result: Dice^mask(all)=0.366, moderate=0.512, small=0.250 (n=100).
# Loaded as iter-0 candidate via `--seed-code` on vtos.runner.seg_orchestrator.
# Anything the search proposes from iter 1 onward should aim to beat this.
bboxes = detect(target, threshold=0.30)
polygons = segment(bboxes)
final_bboxes, final_polygons = bboxes, polygons
