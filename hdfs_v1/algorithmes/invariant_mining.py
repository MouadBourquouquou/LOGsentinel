import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class InvariantMining:

    def __init__(
        self,
        min_r2=0.95,
        max_invariants=20,
        aggregation="mean"
    ):
        self.min_r2 = min_r2
        self.max_invariants = max_invariants
        self.aggregation = aggregation
        self.invariants = []
        self.n_features = None

    def fit(self, X):

        X = np.asarray(X, dtype=float)

        self.n_features = X.shape[1]
        self.invariants = []

        # Chaque caractéristique est prédite
        # à partir de toutes les autres caractéristiques.
        for j in range(self.n_features):

            predictors = [
                k for k in range(self.n_features)
                if k != j
            ]

            X_predictors = X[:, predictors]
            y_target = X[:, j]

            # Ignorer les caractéristiques constantes
            if np.std(y_target) < 1e-12:
                continue

            model = LinearRegression()
            model.fit(X_predictors, y_target)

            y_reconstructed = model.predict(X_predictors)

            r2 = r2_score(y_target, y_reconstructed)

            # Résidus sur les données normales d'entraînement
            residuals = y_target - y_reconstructed

            # Échelle robuste basée sur MAD
            median_res = np.median(residuals)

            mad = np.median(
                np.abs(residuals - median_res)
            )

            sigma = 1.4826 * mad

            # Solution de secours si MAD est nul
            if sigma < 1e-12:
                sigma = np.std(residuals)

            sigma = max(sigma, 1e-12)

            if r2 >= self.min_r2:

                self.invariants.append({
                    "target": j,
                    "predictors": predictors,
                    "model": model,
                    "r2": r2,
                    "sigma": sigma
                })

        # Trier les invariants par R² décroissant
        self.invariants.sort(
            key=lambda inv: inv["r2"],
            reverse=True
        )

        # Garder les meilleurs invariants
        self.invariants = self.invariants[
            :self.max_invariants
        ]

        if not self.invariants:
            raise ValueError(
                "Aucun invariant trouvé. "
                "Essaie de diminuer min_r2."
            )

        print("Nombre d'invariants découverts :",
              len(self.invariants))

        print("R² moyen :",
              round(
                  np.mean([
                      inv["r2"] for inv in self.invariants
                  ]), 4
              ))

        return self

    def score_anomaly(self, X):

        X = np.asarray(X, dtype=float)

        if X.shape[1] != self.n_features:
            raise ValueError("Nombre de caractéristiques incorrect.")

        scores_invariants = []

        for inv in self.invariants:

            j = inv["target"]
            predictors = inv["predictors"]

            model = inv["model"]
            sigma = inv["sigma"]

            # Valeur prédite
            y_pred = model.predict(X[:, predictors])

            # Valeur réelle
            y_real = X[:, j]

            # Résidu normalisé
            residual = np.abs(y_real - y_pred) / sigma

            scores_invariants.append(residual)

        scores_invariants = np.array(scores_invariants)

        # Agrégation des violations
        if self.aggregation == "mean":
            scores = np.mean(scores_invariants, axis=0)

        elif self.aggregation == "max":
            scores = np.max(scores_invariants, axis=0)

        else:
            raise ValueError(
                "Aggregation doit être 'mean' ou 'max'."
            )

        return scores


# Fonction de score utilisée dans le notebook
def score(model, X):
    return model.score_anomaly(X)