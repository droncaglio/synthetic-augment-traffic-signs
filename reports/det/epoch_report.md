# Convergência por run (épocas / steps / loss)

Runs com probe val-on (best epoch por mAP de val):

| arm | seed | epochs | best_epoch | mAP_best | mAP_last | gap |
|---|--:|--:|--:|--:|--:|--:|
| zero_aug | 0 | 25 | 25 | 0.6864 | 0.6864 | 0.0 |
| zero_aug | 1 | 25 | 25 | 0.6856 | 0.6856 | 0.0 |
| zero_aug | 2 | 25 | 25 | 0.6812 | 0.6812 | 0.0 |
| zero_aug | 3 | 25 | 25 | 0.6904 | 0.6904 | 0.0 |
| zero_aug | 4 | 25 | 25 | 0.6889 | 0.6889 | 0.0 |
| zero_aug | 5 | 25 | 25 | 0.6994 | 0.6994 | 0.0 |
| zero_aug | 6 | 25 | 25 | 0.6908 | 0.6908 | 0.0 |
| da_only | 0 | 25 | 25 | 0.7648 | 0.7648 | 0.0 |
| da_only | 1 | 25 | 25 | 0.7647 | 0.7647 | 0.0 |
| da_only | 2 | 25 | 25 | 0.7689 | 0.7689 | 0.0 |
| da_only | 3 | 25 | 25 | 0.7675 | 0.7675 | 0.0 |
| da_only | 4 | 25 | 25 | 0.7649 | 0.7649 | 0.0 |
| da_only | 5 | 25 | 25 | 0.7662 | 0.7662 | 0.0 |
| da_only | 6 | 25 | 25 | 0.7684 | 0.7684 | 0.0 |
| real_duplicate | 0 | 21 | 21 | 0.7729 | 0.7729 | 0.0 |
| real_duplicate | 1 | 21 | 21 | 0.7725 | 0.7725 | 0.0 |
| real_duplicate | 2 | 21 | 21 | 0.7657 | 0.7657 | 0.0 |
| real_duplicate | 3 | 21 | 21 | 0.7672 | 0.7672 | 0.0 |
| real_duplicate | 4 | 21 | 21 | 0.7739 | 0.7739 | 0.0 |
| real_duplicate | 5 | 21 | 21 | 0.7691 | 0.7691 | 0.0 |
| real_duplicate | 6 | 21 | 21 | 0.7751 | 0.7751 | 0.0 |
| bg_photometric | 0 | 21 | 21 | 0.7599 | 0.7599 | 0.0 |
| bg_photometric | 1 | 21 | 21 | 0.7715 | 0.7715 | 0.0 |
| bg_photometric | 2 | 21 | 21 | 0.7679 | 0.7679 | 0.0 |
| bg_photometric | 3 | 21 | 21 | 0.7647 | 0.7647 | 0.0 |
| bg_photometric | 4 | 21 | 21 | 0.7635 | 0.7635 | 0.0 |
| bg_photometric | 5 | 21 | 21 | 0.7758 | 0.7758 | 0.0 |
| bg_photometric | 6 | 21 | 21 | 0.7644 | 0.7644 | 0.0 |
| copy_paste | 0 | 21 | 21 | 0.7643 | 0.7643 | 0.0 |
| copy_paste | 1 | 21 | 21 | 0.771 | 0.771 | 0.0 |
| copy_paste | 2 | 21 | 21 | 0.7698 | 0.7698 | 0.0 |
| copy_paste | 3 | 21 | 21 | 0.764 | 0.764 | 0.0 |
| copy_paste | 4 | 21 | 21 | 0.7745 | 0.7745 | 0.0 |
| copy_paste | 5 | 21 | 21 | 0.7656 | 0.7656 | 0.0 |
| copy_paste | 6 | 21 | 21 | 0.7707 | 0.7707 | 0.0 |
| diffusion_bg | 0 | 21 | 21 | 0.7656 | 0.7656 | 0.0 |
| diffusion_bg | 1 | 21 | 21 | 0.7653 | 0.7653 | 0.0 |
| diffusion_bg | 2 | 21 | 21 | 0.763 | 0.763 | 0.0 |
| diffusion_bg | 3 | 21 | 21 | 0.7703 | 0.7703 | 0.0 |
| diffusion_bg | 4 | 21 | 21 | 0.7714 | 0.7714 | 0.0 |
| diffusion_bg | 5 | 21 | 21 | 0.7738 | 0.7738 | 0.0 |
| diffusion_bg | 6 | 21 | 21 | 0.7721 | 0.7721 | 0.0 |

Runs val-off (proxy de platô = queda da loss no último 20%; ~0 = achatou):

| arm | seed | epochs | steps | dev | loss_first | loss_last | tail_drop_% |
|---|--:|--:|--:|--:|--:|--:|--:|
