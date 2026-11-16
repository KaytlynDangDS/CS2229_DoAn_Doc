"""Thực nghiệm tái hiện (thu nhỏ) Mục 6 của Hardt, Recht, Singer (2016).

Ý tưởng giống bài báo: tạo hai tập huấn luyện S, S' chỉ khác nhau đúng một mẫu,
chạy SGD trên hai tập với CÙNG điểm khởi tạo và CÙNG thứ tự duyệt mẫu (coupling),
rồi đo
  * delta_t = ||w_t - w'_t||  (khoảng cách tham số, đại diện cho độ ổn định),
  * generalization gap = loss/error trên tập test - loss/error trên tập train.

Dữ liệu: bộ chữ số viết tay 8x8 của scikit-learn (load_digits, có sẵn offline).
Mô hình: hồi quy logistic đa lớp (loss lồi) và mạng một lớp ẩn tanh (loss không lồi).
SGD: minibatch 1, duyệt theo hoán vị ngẫu nhiên mỗi epoch, learning rate cố định.

Chạy:  python code/stability_experiments.py
Kết quả: các hình PDF trong graphics/ và bảng số liệu graphics/results.json
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "graphics")
os.makedirs(OUT, exist_ok=True)

N_TRAIN = 1000          # số mẫu huấn luyện n
EPOCHS = 30
LOG_EVERY = 100         # đo delta_t sau mỗi 100 bước, như bài báo
N_PAIRS = 5             # số cặp (S, S') để lấy trung bình
HIDDEN = 100


# ---------------------------------------------------------------- dữ liệu
def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    # n+1 mẫu đầu: n mẫu cho S và 1 mẫu "dự phòng" để tạo S'
    return X[: N_TRAIN + 1], y[: N_TRAIN + 1], X[N_TRAIN + 1:], y[N_TRAIN + 1:]


def make_pair(Xp, yp, rng):
    """Tạo S và S' khác nhau đúng một mẫu (giống thủ tục Mục 6 của bài báo).

    Bỏ một mẫu ngẫu nhiên ra làm mẫu dự phòng -> S.
    Thay một mẫu ngẫu nhiên của S bằng mẫu dự phòng -> S'.
    Trả về thêm vị trí k của mẫu khác biệt.
    """
    spare = rng.integers(len(yp))
    keep = np.delete(np.arange(len(yp)), spare)
    XS, yS = Xp[keep].copy(), yp[keep].copy()
    k = rng.integers(N_TRAIN)
    XS2, yS2 = XS.copy(), yS.copy()
    XS2[k], yS2[k] = Xp[spare], yp[spare]
    return (XS, yS), (XS2, yS2), k


# ---------------------------------------------------------------- mô hình
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Logistic:
    """Hồi quy logistic đa lớp: loss cross-entropy lồi theo tham số."""

    convex = True

    def init(self, rng):
        return [np.zeros((64, 10)), np.zeros(10)]

    def forward(self, p, X):
        return softmax(X @ p[0] + p[1])

    def grad(self, p, x, y):
        pr = softmax(x @ p[0] + p[1])
        pr[y] -= 1.0
        return [np.outer(x, pr), pr]


class MLP:
    """Mạng một lớp ẩn tanh: loss không lồi theo tham số."""

    convex = False

    def init(self, rng):
        return [rng.normal(0, 1 / 8, (64, HIDDEN)), np.zeros(HIDDEN),
                rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, 10)), np.zeros(10)]

    def forward(self, p, X):
        h = np.tanh(X @ p[0] + p[1])
        return softmax(h @ p[2] + p[3])

    def grad(self, p, x, y):
        h = np.tanh(x @ p[0] + p[1])
        pr = softmax(h @ p[2] + p[3])
        pr[y] -= 1.0
        gW2, gb2 = np.outer(h, pr), pr
        dh = (p[2] @ pr) * (1 - h ** 2)
        return [np.outer(x, dh), dh, gW2, gb2]


def metrics(model, p, X, y):
    pr = model.forward(p, X)
    loss = -np.mean(np.log(pr[np.arange(len(y)), y] + 1e-12))
    err = np.mean(pr.argmax(1) != y)
    return loss, err


def dist(p, q):
    return float(np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(p, q))))


# ---------------------------------------------------------------- một lần chạy cặp
def run_pair(model, S, S2, k, Xte, yte, alpha, seed, weight_decay=0.0,
             force_pos=None, epochs=EPOCHS):
    """Chạy SGD song song trên S và S' với cùng khởi tạo và cùng thứ tự duyệt.

    force_pos: nếu khác None, đặt mẫu khác biệt k vào vị trí force_pos
    ("first" hoặc "last") của hoán vị trong epoch đầu tiên.
    """
    rng = np.random.default_rng(seed)
    p = model.init(rng)
    q = [a.copy() for a in p]
    steps, deltas, epoch_log = [], [], []
    t = 0
    for ep in range(epochs):
        order = rng.permutation(N_TRAIN)
        if ep == 0 and force_pos is not None:
            order = order[order != k]
            order = np.concatenate([[k], order]) if force_pos == "first" else np.concatenate([order, [k]])
        for i in order:
            gp = model.grad(p, S[0][i], S[1][i])
            gq = model.grad(q, S2[0][i], S2[1][i])
            for a, g in zip(p, gp):
                a *= (1 - alpha * weight_decay)
                a -= alpha * g
            for a, g in zip(q, gq):
                a *= (1 - alpha * weight_decay)
                a -= alpha * g
            t += 1
            if t % LOG_EVERY == 0:
                steps.append(t)
                deltas.append(dist(p, q))
        tr_loss, tr_err = metrics(model, p, *S)
        te_loss, te_err = metrics(model, p, Xte, yte)
        epoch_log.append(dict(epoch=ep + 1, train_loss=tr_loss, test_loss=te_loss,
                              train_err=tr_err, test_err=te_err))
    return np.array(steps), np.array(deltas), epoch_log


def averaged(model, data, alpha, **kw):
    Xp, yp, Xte, yte = data
    D, logs = [], []
    for s in range(N_PAIRS):
        rng = np.random.default_rng(100 + s)
        S, S2, k = make_pair(Xp, yp, rng)
        steps, d, log = run_pair(model, S, S2, k, Xte, yte, alpha, seed=200 + s, **kw)
        D.append(d)
        logs.append(log)
    gap_loss = np.mean([[e["test_loss"] - e["train_loss"] for e in lg] for lg in logs], axis=0)
    gap_err = np.mean([[e["test_err"] - e["train_err"] for e in lg] for lg in logs], axis=0)
    tr_err = np.mean([[e["train_err"] for e in lg] for lg in logs], axis=0)
    te_err = np.mean([[e["test_err"] for e in lg] for lg in logs], axis=0)
    return dict(steps=steps, delta=np.mean(D, axis=0), gap_loss=gap_loss, gap_err=gap_err,
                train_err=tr_err, test_err=te_err)


# ---------------------------------------------------------------- vẽ hình
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "figure.dpi": 150})
BLUES = ["#9ecae1", "#4292c6", "#08519c"]


def fig_lr(res, alphas, name, title, logy=False):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ep = np.arange(1, EPOCHS + 1)
    for a, c in zip(alphas, BLUES):
        r = res[a]
        ax[0].plot(r["steps"] / N_TRAIN, r["delta"], color=c, label=f"α = {a}")
        ax[1].plot(ep, r["gap_loss"], color=c, label=f"α = {a}")
        ax[2].plot(ep, r["gap_err"] * 100, color=c, label=f"α = {a}")
    ax[0].set(xlabel="Epoch", ylabel="‖w_t − w′_t‖", title="Khoảng cách tham số")
    if logy:
        ax[0].set_yscale("log")
        ax[0].set_title("Khoảng cách tham số (thang log)")
    ax[1].set(xlabel="Epoch", ylabel="loss test − loss train", title="Generalization gap (cross-entropy)")
    ax[2].set(xlabel="Epoch", ylabel="điểm %", title="Generalization gap (tỉ lệ lỗi)")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend(frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_data()
    results = {}

    # Thí nghiệm 1 và 2: ảnh hưởng của learning rate (lồi và không lồi)
    alphas = [0.025, 0.05, 0.1]
    for model, tag, title in [(Logistic(), "convex", "Hồi quy logistic (loss lồi)"),
                              (MLP(), "nonconvex", "Mạng một lớp ẩn (loss không lồi)")]:
        res = {a: averaged(model, data, a) for a in alphas}
        fig_lr(res, alphas, f"exp_{tag}_lr.pdf", title, logy=not model.convex)
        results[tag] = {str(a): dict(delta_final=float(r["delta"][-1]),
                                     delta_epoch10=float(r["delta"][10 * N_TRAIN // LOG_EVERY - 1]),
                                     gap_loss_final=float(r["gap_loss"][-1]),
                                     gap_err_final=float(r["gap_err"][-1]),
                                     train_err_final=float(r["train_err"][-1]),
                                     test_err_final=float(r["test_err"][-1]))
                        for a, r in res.items()}
        if tag == "nonconvex":
            # Thí nghiệm 3: khoảng cách tham số và gap đi cùng nhau
            r = res[0.05]
            ep = np.arange(1, EPOCHS + 1)
            d_ep = r["delta"][N_TRAIN // LOG_EVERY - 1::N_TRAIN // LOG_EVERY]
            fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
            ax1.plot(ep, r["gap_loss"], color="#08519c", lw=2, label="Gap cross-entropy (trục trái)")
            ax1.set(xlabel="Epoch", ylabel="loss test − loss train")
            ax2 = ax1.twinx()
            ax2.plot(ep, d_ep, color="#e6550d", ls="--", lw=2, label="‖w_t − w′_t‖ (trục phải)")
            ax2.set_ylabel("Khoảng cách tham số")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right")
            ax1.grid(alpha=0.3)
            ax1.set_title("Mạng một lớp ẩn, α = 0,05: gap và khoảng cách tham số")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "exp_gap_vs_distance.pdf"), bbox_inches="tight")
            plt.close(fig)
            corr = float(np.corrcoef(d_ep, r["gap_loss"])[0, 1])
            results["corr_delta_gaploss_mlp_0.05"] = corr

    # Thí nghiệm 4: burn-in, mẫu khác biệt ở đầu hay cuối epoch đầu tiên
    Xp, yp, Xte, yte = data
    model = MLP()
    curves = {}
    for pos in ["first", "last"]:
        D = []
        for s in range(N_PAIRS):
            rng = np.random.default_rng(100 + s)
            S, S2, k = make_pair(Xp, yp, rng)
            steps, d, _ = run_pair(model, S, S2, k, Xte, yte, 0.05, seed=200 + s,
                                   force_pos=pos, epochs=10)
            D.append(d)
        curves[pos] = (steps, np.mean(D, axis=0))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(curves["first"][0] / N_TRAIN, curves["first"][1], color="#e6550d",
            label="Mẫu khác biệt ở bước đầu tiên")
    ax.plot(curves["last"][0] / N_TRAIN, curves["last"][1], color="#08519c",
            label="Mẫu khác biệt ở cuối epoch đầu")
    ax.set(xlabel="Epoch", ylabel="‖w_t − w′_t‖", title="Burn-in: thời điểm gặp mẫu khác biệt")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp_burnin.pdf"), bbox_inches="tight")
    plt.close(fig)
    results["burnin"] = {pos: dict(delta_epoch1=float(c[1][N_TRAIN // LOG_EVERY - 1]),
                                   delta_epoch10=float(c[1][-1])) for pos, c in curves.items()}

    # Thí nghiệm 5: weight decay
    wd = {}
    for mu in [0.0, 0.01]:
        r = averaged(MLP(), data, 0.05, weight_decay=mu)
        wd[mu] = r
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = np.arange(1, EPOCHS + 1)
    for mu, c in zip([0.0, 0.01], ["#9ecae1", "#08519c"]):
        ax[0].plot(wd[mu]["steps"] / N_TRAIN, wd[mu]["delta"], color=c, label=f"μ = {mu}")
        ax[1].plot(ep, wd[mu]["gap_loss"], color=c, label=f"μ = {mu}")
    ax[0].set(xlabel="Epoch", ylabel="‖w_t − w′_t‖", title="Khoảng cách tham số")
    ax[1].set(xlabel="Epoch", ylabel="loss test − loss train", title="Generalization gap")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend(frameon=False)
    fig.suptitle("Weight decay, mạng một lớp ẩn, α = 0,05", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp_weight_decay.pdf"), bbox_inches="tight")
    plt.close(fig)
    results["weight_decay"] = {str(mu): dict(delta_final=float(r["delta"][-1]),
                                             gap_loss_final=float(r["gap_loss"][-1]),
                                             train_err_final=float(r["train_err"][-1]),
                                             test_err_final=float(r["test_err"][-1]))
                               for mu, r in wd.items()}

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
