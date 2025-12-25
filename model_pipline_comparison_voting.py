import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import warnings
warnings.filterwarnings("ignore")

from Preprocess_Data import load_and_combine


def show_confusion_matrix(
    cnf_matrix,
    class_labels=(0, 1),
    title="Confusion Matrix",
    save_path=None,
    show=True,
    dpi=220
):
    #Plot a confusion matrix, with an option to save it as an image
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    ax.imshow(cnf_matrix, cmap=plt.cm.YlGn, alpha=0.7)

    # avoiding overlapping(x-axis label)
    fig.suptitle(title, fontsize=14, y=0.98)

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=12, labelpad=10)

    ax.set_xticks(range(0, len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45)
    ax.set_yticks(range(0, len(class_labels)))
    ax.set_yticklabels(class_labels)

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # data
    for (i, j), v in np.ndenumerate(cnf_matrix):
        ax.text(j, i, int(v), va="center", ha="center", fontsize=12)

    # leave some space for suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    plot_cm=True,
    cm_save_path=None,
    cm_show=True
):
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\n[{name}]")
    print(f"train accuracy: {train_acc:.6f}")
    print(f"test  accuracy: {test_acc:.6f}")

    if plot_cm:
        y_pred = model.predict(X_test)
        cnf = metrics.confusion_matrix(y_test, y_pred)
        show_confusion_matrix(
            cnf,
            class_labels=(0, 1),
            title=f"{name} (Test)",
            save_path=cm_save_path,
            show=cm_show
        )


def grid_search(pipe, param_grid, X_train, y_train, cv, scoring="accuracy", n_jobs=-1, verbose=1):
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)
    return gs


def randomized_search(
    pipe,
    param_distributions,
    X_train,
    y_train,
    cv,
    scoring="accuracy",
    n_iter=30,
    n_jobs=-1,
    verbose=1,
    random_state=42
):
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
    )
    rs.fit(X_train, y_train)
    return rs


if __name__ == "__main__":
    train_X, test_x, train_y, test_y = load_and_combine()

    # save fig.confussion matrix
    ASSETS_DIR = "assets"
    os.makedirs(ASSETS_DIR, exist_ok=True)

    NUM_COLS = ['SibSp', 'Parch', 'Fare', 'FamilySize', 'is_mother']
    CAT_COLS = [c for c in train_X.columns if c not in NUM_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", "passthrough", CAT_COLS),
        ],
        remainder="drop"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------------------------
    # Logistic Regression (GridSearchCV)
    # ---------------------------
    lr_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    lr_param_grid = {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["liblinear", "lbfgs"],
        "clf__penalty": ["l2"],
    }

    lr_gs = grid_search(lr_pipe, lr_param_grid, train_X, train_y, cv=cv, verbose=0)
    best_lr = lr_gs.best_estimator_
    print("\n[LR] Best CV accuracy:", lr_gs.best_score_)
    print("[LR] Best params:", lr_gs.best_params_)

    evaluate_model(
        "LogisticRegression (tuned)",
        best_lr,
        train_X, train_y,
        test_x, test_y,
        plot_cm=True,
        cm_save_path=os.path.join(ASSETS_DIR, "cm_lr_test.png"),
        cm_show=False
    )

    feature_names = NUM_COLS + CAT_COLS
    lr_model = best_lr.named_steps["clf"]
    coef_df = pd.DataFrame({
        "coef": lr_model.coef_.ravel(),
        "feature": feature_names
    }).sort_values("coef", ascending=False)
    print("\nTop positive coefficients (LR):")
    print(coef_df.head(10))
    print("\nTop negative coefficients (LR):")
    print(coef_df.tail(10))

    # ---------------------------
    # SVC (GridSearchCV)
    # ---------------------------
    svc_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", SVC()),
    ])

    svc_param_grid = [
        {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10, 100]},
        {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100],
         "clf__gamma": ["scale", "auto", 0.01, 0.1, 1]},
    ]

    svc_gs = grid_search(svc_pipe, svc_param_grid, train_X, train_y, cv=cv, verbose=0)
    best_svc = svc_gs.best_estimator_
    print("\n[SVC] Best CV accuracy:", svc_gs.best_score_)
    print("[SVC] Best params:", svc_gs.best_params_)

    evaluate_model(
        "SVC (tuned)",
        best_svc,
        train_X, train_y,
        test_x, test_y,
        plot_cm=True,
        cm_save_path=os.path.join(ASSETS_DIR, "cm_svc_test.png"),
        cm_show=False
    )

    # ---------------------------
    # KNN (GridSearchCV)
    # ---------------------------
    knn_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", KNeighborsClassifier()),
    ])

    knn_param_grid = {
        "clf__n_neighbors": list(range(3, 31, 2)),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],  # 1: Manhattan, 2: Euclidean
    }

    knn_gs = grid_search(knn_pipe, knn_param_grid, train_X, train_y, cv=cv, verbose=0)
    best_knn = knn_gs.best_estimator_
    print("\n[KNN] Best CV accuracy:", knn_gs.best_score_)
    print("[KNN] Best params:", knn_gs.best_params_)

    evaluate_model(
        "KNN (tuned)",
        best_knn,
        train_X, train_y,
        test_x, test_y,
        plot_cm=True,
        cm_save_path=os.path.join(ASSETS_DIR, "cm_knn_test.png"),
        cm_show=False
    )

    # ---------------------------
    # Decision Tree (GridSearchCV)
    # ---------------------------
    dt_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ])

    dt_param_grid = {
        "clf__max_depth": [None, 3, 5, 7, 9, 12],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
    }

    dt_gs = grid_search(dt_pipe, dt_param_grid, train_X, train_y, cv=cv, verbose=0)
    best_dt = dt_gs.best_estimator_
    print("\n[DT] Best CV accuracy:", dt_gs.best_score_)
    print("[DT] Best params:", dt_gs.best_params_)

    evaluate_model(
        "DecisionTree (tuned)",
        best_dt,
        train_X, train_y,
        test_x, test_y,
        plot_cm=True,
        cm_save_path=os.path.join(ASSETS_DIR, "cm_dt_test.png"),
        cm_show=False
    )

    # ---------------------------
    # Random Forest (RandomizedSearchCV)
    # ---------------------------
    rf_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(random_state=42)),
    ])

    rf_param_dist = {
        "clf__n_estimators": [100, 200, 400, 800],
        "clf__max_depth": [None, 3, 5, 7, 9, 12],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", None],
    }

    rf_rs = randomized_search(
        rf_pipe,
        rf_param_dist,
        train_X,
        train_y,
        cv=cv,
        n_iter=30,
        verbose=0,
        random_state=42,
    )
    best_rf = rf_rs.best_estimator_
    print("\n[RF] Best CV accuracy:", rf_rs.best_score_)
    print("[RF] Best params:", rf_rs.best_params_)

    evaluate_model(
        "RandomForest (tuned)",
        best_rf,
        train_X, train_y,
        test_x, test_y,
        plot_cm=True,
        cm_save_path=os.path.join(ASSETS_DIR, "cm_rf_test.png"),
        cm_show=False
    )

    # ---------------------------
    # Voting Ensemble
    # ---------------------------
    votingcf = VotingClassifier(
        estimators=[
            ("lr", best_lr),
            ("svm", best_svc),
            ("knn", best_knn),
            ("dt", best_dt),
            ("rf", best_rf),
        ],
        voting="hard",
        n_jobs=-1,
    )

    votingcf.fit(train_X, train_y)

    evaluate_model(
        "VotingClassifier (tuned base models)",
        votingcf,
        train_X, train_y,
        test_x, test_y,
        plot_cm=True,
        cm_save_path=os.path.join(ASSETS_DIR, "cm_voting_test.png"),
        cm_show=True
    )

    print(f"\nSaved confusion matrix images to: ./{ASSETS_DIR}/")
    print(f"- {ASSETS_DIR}/cm_lr_test.png")
    print(f"- {ASSETS_DIR}/cm_voting_test.png")
