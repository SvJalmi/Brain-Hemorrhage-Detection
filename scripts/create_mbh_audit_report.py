from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(r"C:\Users\shrey\OneDrive\Desktop\Brain_Hemorrhage_Project")
REPORT_PATH = PROJECT_ROOT / "dataset_check" / "MBH_SEG25_MASTER_AUDIT_REPORT.txt"

report = f"""
============================================================
MBH-Seg25 MASTER DATASET AUDIT REPORT
============================================================

Audit date:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset:
MBH-Seg25 training dataset

Source location:
C:\\Users\\shrey\\Downloads\\MBH_Train_2025_voxel-label (1)

IMPORTANT:
This report documents read-only validation performed on the
original dataset. No image, annotation, or case folder was
modified, renamed, deleted, moved, resampled, clipped, or
converted during these checks.


============================================================
1. DATASET STRUCTURE
============================================================

Case folders:
192

Cases with image.nii.gz:
192

Cases with exactly one image.nii.gz:
192

Cases with zero image.nii.gz:
0

Cases with more than one image.nii.gz:
0

Total CT volumes:
192

Conclusion:
Every case contains exactly one CT volume named image.nii.gz.


============================================================
2. ANNOTATION FILE STRUCTURE
============================================================

Total annotation files:
471

Annotation file counts:

label_annot_1.nii.gz: 120
label_annot_2.nii.gz: 115
label_annot_3.nii.gz: 192
label_annot_4.nii.gz: 44

Annotation count per case:

2 annotations: 111 cases
3 annotations: 75 cases
4 annotations: 6 cases

Conclusion:
The dataset is multi-rater. Annotation number identifies the
annotator/rater; it is NOT the hemorrhage class.


============================================================
3. SEMANTIC LABEL ENCODING
============================================================

Official semantic mapping used during validation:

0 = background
1 = epidural
2 = intraparenchymal
3 = intraventricular
4 = subarachnoid
5 = subdural

Global annotation label integrity:

Annotation files checked:
471

Unexpected labels outside 0-5:
0


============================================================
4. IMAGE / ANNOTATION GEOMETRY
============================================================

All 471 annotation volumes were checked against their
corresponding image volume.

Shape mismatches:
0

All annotation volumes therefore have the same voxel-array
dimensions as their corresponding CT image.

Affine validation:

A strict numerical comparison identified small affine
differences in a limited number of annotation files.

Maximum translation-component difference:
0.0

Maximum voxel-spacing difference:
approximately 4.49e-07

Largest observed full-affine component difference:
approximately 0.000508

These differences were small numerical/orientation-matrix
differences without meaningful translation or voxel-spacing
disagreement.

No resampling or affine correction was performed.


============================================================
5. CT VOLUME INTEGRITY
============================================================

CT volumes checked:
192

Missing/unreadable CT volumes:
0

Empty/zero CT volumes:
0

Constant CT volumes:
0

All CT volumes have dimensions:
512 x 512 x Z

The number of slices varies between cases.


============================================================
6. CT SLICE-COUNT DISTRIBUTION
============================================================

Observed slice counts include:

22, 23, 25, 26, 27, 28, 29, 30, 31, 32,
33, 34, 35, 36, 37, 38, 39, 40, 42, 44,
46, 58

Therefore the dataset contains variable-depth 3D CT volumes.


============================================================
7. CT INTENSITY RANGE
============================================================

The dataset does not uniformly remain within the conventional
-2048 to 3071 CT intensity range.

Across 192 CT volumes:

Cases with values below -2048:
50

Cases entirely within -2048 to 3071:
122

Cases with values above 3071:
27

Cases with values above 10000:
21

Cases with values below -3000:
50


------------------------------------------------------------
7A. LOW INTENSITY / -3024 INVESTIGATION
------------------------------------------------------------

A recurring value of -3024 was observed in multiple cases.

One representative high-percentage case:

ID_30d19531_ID_cf607a5cd7

Shape:
512 x 512 x 30

-3024 voxels:
2,819,780

Percentage:
35.8554%

The value -3024 occurred in all 30 slices.

No slice contained more than 50% -3024 voxels.

The -3024 voxels spanned the full array bounding box:

Rows:
0 to 511

Columns:
0 to 511

Slices:
0 to 29

Annotation comparison for this case:

Annotation files:
2

Foreground annotation voxels:
213,166

-3024 voxels overlapping annotation foreground:
9

-3024 voxels outside annotation foreground:
2,819,771

Percentage of -3024 voxels overlapping annotation:
0.000319%

Conclusion:
The -3024 value overwhelmingly occurs outside annotated
hemorrhage foreground in the investigated case.

However, 9 overlapping voxels were observed, so -3024 should
NOT be universally hard-coded as background without further
dataset documentation.

No clipping, replacement, or reinterpretation of -3024 was
performed.


------------------------------------------------------------
7B. HIGH INTENSITY INVESTIGATION
------------------------------------------------------------

A representative case with the highest observed proportion
of values above 3071 was investigated:

ID_85f42dd5_ID_a0ef1a9eaf

Shape:
512 x 512 x 36

Maximum:
29,833

Voxels above 3071:
10,073

Percentage above 3071:
0.106737%

Voxels above 10000:
2,001

Voxels above 20000:
63

Slices containing values above 3071:
8

No slice contained more than 1% of its voxels above 3071.

Annotation comparison:

Annotation files:
3

Foreground annotation voxels:
3,158

High-intensity voxels overlapping annotation foreground:
0

High-intensity voxels outside annotation foreground:
10,073

Percentage overlapping annotation:
0%

Spatial location:

Rows:
175 to 304

Columns:
6 to 108

Slices:
0 to 7

Of the 10,073 high-intensity voxels:

Within 20 pixels of image border:
973 (9.6595%)

Not within 20 pixels of image border:
9,100 (90.3405%)

Conclusion:
The high-intensity values are sparse and spatially localized
in the investigated case. They do not overlap the annotated
hemorrhage foreground in that case.

Their exact physical/semantic origin has not been established.

No clipping or replacement was performed.


============================================================
8. VOXEL-LEVEL LABEL DISTRIBUTION
============================================================

Across all 471 annotation files:

Label 0 / background:
3,951,221,669 voxels

Label 1 / epidural:
398,861 voxels

Label 2 / intraparenchymal:
5,623,888 voxels

Label 3 / intraventricular:
1,783,287 voxels

Label 4 / subarachnoid:
1,638,810 voxels

Label 5 / subdural:
3,212,909 voxels

Unexpected labels:
None


============================================================
9. CASE-LEVEL SUBTYPE PRESENCE
============================================================

Number of unique cases containing each hemorrhage subtype:

Epidural:
27 / 192 cases

Intraparenchymal:
138 / 192 cases

Intraventricular:
122 / 192 cases

Subarachnoid:
122 / 192 cases

Subdural:
100 / 192 cases

These categories overlap because individual CT cases may contain
multiple hemorrhage types.


============================================================
10. CASE-LEVEL HEMORRHAGE COMBINATIONS
============================================================

Unique hemorrhage combinations:
26

Observed combinations range from single-subtype cases through
cases containing all five hemorrhage subtypes.

Important examples:

E + I + V + SA + SD:
5 cases

E + I + V + SA:
2 cases

E + I + V + SD:
1 case

E + I + SA + SD:
2 cases

E + V + SA + SD:
1 case

I + V + SA + SD:
32 cases

I + V + SA:
29 cases

I + V:
22 cases

I + SA + SD:
15 cases

V + SA + SD:
10 cases

E only:
4 cases

I only:
8 cases

V only:
6 cases

SA only:
3 cases

SD only:
10 cases

Conclusion:
MBH-Seg25 is a multi-subtype dataset. A single case can contain
multiple hemorrhage categories.


============================================================
11. INTER-RATER FOREGROUND DICE
============================================================

Annotator 1 vs Annotator 3:

Cases:
120

Minimum Dice:
0.0000

Median Dice:
0.7609

Mean Dice:
0.7173

Maximum Dice:
0.9509

Dice = 0:
2

Dice = 1:
0


Annotator 2 vs Annotator 3:

Cases:
115

Minimum Dice:
0.0214

Median Dice:
0.7414

Mean Dice:
0.6906

Maximum Dice:
0.9405

Dice = 0:
0

Dice = 1:
0


Annotator 3 vs Annotator 4:

Cases:
44

Minimum Dice:
0.1893

Median Dice:
0.6639

Mean Dice:
0.6293

Maximum Dice:
0.9135

Dice = 0:
0

Dice = 1:
0

Conclusion:
The annotations demonstrate substantial but imperfect
inter-rater agreement, consistent with genuine multi-rater
segmentation variability.

Annotator 3 was NOT treated as ground truth.


============================================================
12. ANNOTATOR SUBTYPE-SET AGREEMENT
============================================================

Cases checked:
192

Cases with annotator label-set disagreement:
86

Cases without subtype-set disagreement:
106

Percentage with disagreement:
44.8%

Percentage without disagreement:
55.2%


Per-subtype case-level presence disagreement:

Epidural:
16 cases

Intraparenchymal:
19 cases

Intraventricular:
21 cases

Subarachnoid:
26 cases

Subdural:
33 cases

These categories overlap.

Conclusion:
Annotators sometimes disagree about which hemorrhage subtypes
are present in a case. This supports preserving the individual
rater annotations rather than assuming one annotation is
universally correct.


============================================================
13. REPRESENTATIVE ANNOTATION OBSERVATIONS
============================================================

Case:
ID_98450f26_ID_8d66ffdc94

All four annotation volumes have matching shape and essentially
matching geometry.

Observed label sets:

annot_1:
0, 3, 4, 5

annot_2:
0, 4

annot_3:
0, 4

annot_4:
0, 4

The masks are not identical and represent rater differences.

Two empty annotation examples were also observed:

ID_8d914da6_ID_b4eb9c8234
label_annot_1:
empty

ID_d039497d_ID_da6cc2290f
label_annot_1:
empty

These empty annotation files were preserved and not deleted.


============================================================
14. DATASET VALIDATION DECISIONS
============================================================

The following actions were NOT performed:

- No files deleted
- No files renamed
- No files moved
- No files copied
- No NIfTI files converted
- No image resampling
- No affine correction
- No intensity clipping
- No intensity replacement
- No annotation merging
- No annotation deletion
- No manual label correction


============================================================
15. OVERALL ASSESSMENT
============================================================

MBH-Seg25 training data contains:

192 CT volumes
471 multi-rater annotation volumes
5 semantic hemorrhage classes plus background

The dataset is structurally consistent with a multi-rater,
multi-class 3D brain hemorrhage segmentation dataset.

The principal characteristics requiring consideration during
future preprocessing/model development are:

1. Variable number of CT slices per volume.
2. Multiple expert annotations per case.
3. Meaningful inter-rater segmentation variability.
4. Multi-hemorrhage cases are common.
5. Epidural hemorrhage is less prevalent at the case level.
6. CT intensity values are not uniformly restricted to
   -2048 to 3071.
7. The recurring -3024 value can occupy a substantial fraction
   of some volumes.
8. Sparse high-intensity values above 3071 occur in some cases.
9. Original annotations should be preserved for future
   consensus/reference-mask decisions.

No evidence from the performed checks requires deleting or
repairing the original MBH-Seg25 training data.

============================================================
END OF REPORT
============================================================
"""

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report, encoding="utf-8")

print("MBH-Seg25 master audit report created.")
print("Path:", REPORT_PATH)
print("Size:", REPORT_PATH.stat().st_size, "bytes")