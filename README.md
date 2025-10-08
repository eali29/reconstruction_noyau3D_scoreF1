# reconstruction_noyau3D_scoreF1

ÉTAPE 1 : CRÉER UN ENVIRONNEMENT DÉDIÉ

conda create -n stardist_env python=3.10 -y
conda activate stardist_env

ÉTAPE 2 : INSTALLER LES PACKAGES REQUIS
conda install numpy scipy matplotlib scikit-learn -y

a) StarDist et ses dépendances

StarDist est distribué via pip (pas conda) :

pip install stardist csbdeep
pip install tensorflow

b) Kalman filter
pip install filterpy

ÉTAPE 3 : VÉRIFIER INSTALLATION
Lance Python depuis ton environnement et essaie :

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from stardist.models import StarDist3D, StarDist2D
from csbdeep.utils import normalize
from sklearn.metrics import f1_score
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
print("✅ Tout est installé correctement !")


ÉTAPE 4 : EXÉCUTER SCRIPT CI-DESSOUS

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from stardist.models import StarDist3D, StarDist2D
from csbdeep.utils import normalize
from sklearn.metrics import f1_score
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# ------------------------ A) Création volume 3D ------------------------
shape = (20, 64, 64)  # (Z, Y, X)
volume = np.zeros(shape)
z0, y0, x0 = 10, 32, 32
radius = 5
Z, Y, X = np.indices(shape)
mask = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2 < radius**2
volume[mask] = 1.0

psf_sigma = (1.5, 0.7, 0.7)
blurred = gaussian_filter(volume, sigma=psf_sigma)

# ------------------------ B) Chargement modèles ------------------------
model3d = StarDist3D.from_pretrained('3D_demo')
model2d = StarDist2D.from_pretrained('2D_versatile_fluo')

# ------------------------ C) Paramètres ------------------------
noise_levels = [0.01]      # différentes valeurs de bruit
repeats_list = [50, 70,100]       # différents nombres de répétitions

# ------------------------ D) Boucle sur bruits et répétitions ------------------------
for sigma_noise in noise_levels:
    for n_repeats in repeats_list:
        print(f"\n=== Bruit = {sigma_noise}, Répétitions = {n_repeats} ===")

        f1_scores_3d = []
        f1_scores_2d = []

        # pour exemple visuel (on prendra la dernière répétition)
        example_noisy = example_pred_mask3d = example_pred_mask2d = None
        example_centers2d = example_centers3d = example_kf_smoothed = None

        for rep in range(n_repeats):
            noisy = blurred + np.random.normal(0, sigma_noise, size=blurred.shape)
            noisy = np.clip(noisy, 0, 1)
            gt_mask = (volume > 0).astype(int)

            # ---- StarDist 3D ----
            img_norm3d = normalize(noisy, 1, 99.8, axis=(0,1,2))
            labels3d, _ = model3d.predict_instances(img_norm3d)
            pred_mask3d = (labels3d > 0).astype(int)
            f1_3d = f1_score(gt_mask.flatten(), pred_mask3d.flatten())
            f1_scores_3d.append(f1_3d)

            # ---- StarDist 2D + Kalman ----
            centers_x, zs = [], []
            pred_mask2d = np.zeros_like(volume, dtype=int)

            for z in range(shape[0]):
                img2d = noisy[z, :, :]
                img_norm2d = normalize(img2d)
                labels2d, _ = model2d.predict_instances(img_norm2d)
                pred_mask2d[z] = (labels2d > 0).astype(int)

                regions = np.unique(labels2d)[1:]
                if len(regions) > 0:
                    center_candidates = [np.argwhere(labels2d==r).mean(axis=0) for r in regions]
                    center_candidates = np.array(center_candidates)  # (y,x)
                    closest = np.argmin(np.abs(center_candidates[:,0] - y0))
                    centers_x.append(center_candidates[closest][1])
                    zs.append(z)

            # Kalman filter
            kf_smoothed = None
            if centers_x:
                kf = KalmanFilter(dim_x=2, dim_z=1)
                dt = 1.0
                kf.F = np.array([[1, dt],[0,1]])
                kf.H = np.array([[1,0]])
                kf.R = 0.5
                kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
                kf.x = np.array([centers_x[0],0])
                kf.P *= 10
                kf_smoothed = []
                for x_obs in centers_x:
                    kf.predict()
                    kf.update([x_obs])
                    kf_smoothed.append(float(kf.x[0]))

            f1_2d = f1_score(gt_mask.flatten(), pred_mask2d.flatten())
            f1_scores_2d.append(f1_2d)

            # sauvegarde dernière répétition
            example_noisy = noisy.copy()
            example_pred_mask3d = pred_mask3d.copy()
            example_pred_mask2d = pred_mask2d.copy()
            example_centers2d = (zs.copy(), centers_x.copy())
            example_kf_smoothed = (zs.copy(), kf_smoothed.copy() if kf_smoothed is not None else [])

            # calcul centres 3D par plan
            centers3d_perz = []
            for z in range(shape[0]):
                labels_slice = labels3d[z]
                regions3 = np.unique(labels_slice)[1:]
                if len(regions3) > 0:
                    center3 = [np.argwhere(labels_slice==r).mean(axis=0) for r in regions3]
                    center3 = np.array(center3)
                    closest3 = np.argmin(np.abs(center3[:,0] - y0))
                    centers3d_perz.append((z, float(center3[closest3][1])))
                else:
                    centers3d_perz.append((z, np.nan))
            example_centers3d = centers3d_perz

        # ------------------------ Résumé statistiques ------------------------
        mean_3d, median_3d = np.mean(f1_scores_3d), np.median(f1_scores_3d)
        mean_2d, median_2d = np.mean(f1_scores_2d), np.median(f1_scores_2d)

        print(f"StarDist3D → Moyenne={mean_3d:.3f}, Médiane={median_3d:.3f}")
        print(f"StarDist2D+Kalman → Moyenne={mean_2d:.3f}, Médiane={median_2d:.3f}")

        # ------------------------ Montage final ------------------------
        xz_image = example_noisy.max(axis=1)   # (Z,X)

        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        fig.suptitle(f"Bruit={sigma_noise}, Répétitions={n_repeats}", fontsize=14)

        # --- Panel gauche : StarDist3D --- #
        ax = axes[0]
        ax.imshow(xz_image, cmap='gray', origin='lower', aspect='auto')
        ax.set_title("Coupe XZ - StarDist3D")
        ax.set_xlabel("X (colonnes)")
        ax.set_ylabel("Z (plans)")

        centres3_z = [c[0] for c in example_centers3d]
        centres3_x = [c[1] for c in example_centers3d]
        valid3 = ~np.isnan(centres3_x)
        ax.plot(np.array(centres3_x)[valid3], np.array(centres3_z)[valid3],
                '-o', color='yellow', label='centres 3D', markersize=4)
        ax.legend(loc='upper right')

        # --- Panel centre : StarDist2D + Kalman --- #
        ax2 = axes[1]
        ax2.imshow(xz_image, cmap='gray', origin='lower', aspect='auto')
        ax2.set_title("Coupe XZ - StarDist2D + Kalman")
        ax2.set_xlabel("X (colonnes)")
        ax2.set_ylabel("Z (plans)")

        if example_centers2d is not None:
            zs_obs, xs_obs = example_centers2d
            if len(zs_obs) > 0:
                ax2.scatter(xs_obs, zs_obs, c='cyan', s=30, marker='o', label='observés')

        if example_kf_smoothed is not None and example_kf_smoothed[1]:
            zs_kf, xs_kf = example_kf_smoothed
            ax2.plot(xs_kf, zs_kf, '-r', linewidth=2, label='Kalman (lissé)')
            ax2.scatter(xs_kf, zs_kf, c='red', s=20)

        ax2.legend(loc='upper right')

        # --- Panel droit : Boxplot --- #
        ax3 = axes[2]
        data = [f1_scores_3d, f1_scores_2d]
        labels = ["StarDist3D", "StarDist2D+Kalman"]
        bp = ax3.boxplot(data, patch_artist=True, labels=labels, showmeans=True)

        colors = ["lightcoral", "lightblue"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax3.set_ylabel("F1-score")
        ax3.set_title("Distribution des F1-scores")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
