# Doppeldwarf-306 · narration draft

221 words · English · 120-second production budget including credits.

Approved by the owner with af_heart narration. Scene timings are budgets, not measured full-narration timings.

## 1. Which dwarf is this? (0-12 s)

Across Wrocław, bronze dwarves share poses and props. Given a photograph, which individual statue are we looking at? Meet Doppeldwarf-306.

Source: [RESULTS.md - Dataset](../../RESULTS.md).

## 2. Photographs become vectors (12-30 s)

A frozen vision model turns each image into a vector: a numerical visual description. Cosine similarity compares these vectors. Similar directions suggest similar appearance; they do not prove identity. We rank each statue by its best reference photograph, without fine-tuning the model.

Source: [RESULTS.md - Method](../../RESULTS.md).

## 3. One number misses the question (30-47 s)

Now add more candidates. Do lookalikes overwhelm retrieval? Instead of reporting one accuracy number, we repeatedly sample larger candidate pools, always keeping the correct statue. DINOv2's accuracy falls slowly; CLIP's falls faster. The curve is the result.

Source: [RESULTS.md - 2. The headline result: accuracy against candidate-pool size](../../RESULTS.md).

## 4. A weight, rather than a filter (47-69 s)

Surely location solves this? But nearby lookalikes survive a hard geographic filter. In simulation, a soft distance bonus works better. Across 294 placed statues, DINOv2 rises from 93.4 to 98.7 percent for users twenty-five metres away.

Source: [RESULTS.md - 16. The reviewer checks: what the explanations survive](../../RESULTS.md).

## 5. New photographer, new challenge (69-88 s)

A new photographer brings different viewpoints, backgrounds, and lighting. Accuracy drops; fewer reference photographs also contribute. LoFTR matches local details across photographs. Geometric consistency adds evidence to visual similarity, moving a better match up the shortlist.

Source: [RESULTS.md - 9. Is it recognising the statue, or the photographer?; 18. Internal reviews: sensitivity, scarcity and a multi-vector baseline](../../RESULTS.md).

Source: [RESULTS.md - 16. The reviewer checks: what the explanations survive](../../RESULTS.md).

## 6. Look at the gap (88-104 s)

Finally, high similarity need not mean a known statue. Comparing the best two candidates exposes ambiguity. This margin reduces false acceptance in the research; the browser demo uses a separately measured close-call hint.

Source: [RESULTS.md - 16. The reviewer checks: what the explanations survive](../../RESULTS.md).

## 7. Test the conditions (104-112 s)

The lesson: test the pool, the place, and the photographer. Explore the dataset, and try the identifier.

Source: [RESULTS.md - Summary of findings](../../RESULTS.md).

## 8. Credits (112-120 s)

(No narration; hold credits.)

Source: [docs/assets/references.json - images](../../docs/assets/references.json).
