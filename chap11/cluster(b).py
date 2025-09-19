# pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch_cluster import knn_graph, random_walk
from scipy.cluster.hierarchy import linkage, fcluster

# 한글 폰트 설정
plt.rcParams["font.family"] = ["Malgun Gothic"]
plt.rcParams["axes.unicode_minus"] = False


# %%
def load_data(file_path):
    print(f"1. 데이터 로드 중: {file_path}")

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # 수익률 계산
    returns = df.pct_change().dropna()

    return returns


# %%
def create_features(returns, window=60):
    print(f"2. 특성 벡터 생성 중 (window={window}일)...")

    features = []

    for i in range(window, len(returns)):
        period_data = returns.iloc[i - window : i]

        # 특성 계산
        volatility = period_data.std().values  # 변동성
        mean_return = period_data.mean().values  # 평균 수익률

        corr_matrix = period_data.corr().values
        corr_features = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]

        feature_vector = np.concatenate([volatility, mean_return, corr_features])
        features.append(feature_vector)

    features = np.array(features)

    return features


# %%
def random_walk_clustering(features, n_clusters=4):
    print(f"3. Random Walk 클러스터링 실행 중 (k={n_clusters})...")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X = torch.FloatTensor(features_scaled)

    k_neighbors = min(10, X.size(0) - 1)
    edge_index = knn_graph(X, k=k_neighbors)

    walk_embeddings = []
    for start_node in range(X.size(0)):
        node_walks = []
        for _ in range(10):
            walk = random_walk(
                edge_index[0],
                edge_index[1],
                start=torch.tensor([start_node]),
                walk_length=20,
            )
            node_walks.append(walk.float())

        walk_embedding = torch.stack(node_walks).mean(dim=0).flatten()
        walk_embeddings.append(walk_embedding)

    walk_features = torch.stack(walk_embeddings).numpy()
    print(f"   - Random Walk 임베딩 생성 완료 (차원: {walk_features.shape[1]})")

    similarity_matrix = cosine_similarity(walk_features)
    distance_matrix = 1 - similarity_matrix
    linkage_matrix = linkage(distance_matrix, method="ward")
    cluster_ids = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    cluster_ids = cluster_ids - 1  # 0부터 시작하도록 조정

    cluster_centers = []
    for cluster_id in range(n_clusters):
        mask = cluster_ids == cluster_id
        if mask.sum() > 0:
            center = walk_features[mask].mean(axis=0)
            cluster_centers.append(center)
        else:
            cluster_centers.append(walk_features.mean(axis=0))

    cluster_centers = np.array(cluster_centers)

    return torch.LongTensor(cluster_ids), torch.FloatTensor(cluster_centers), scaler

# %%
def optimize_portfolios(returns, clusters, window=60):
    print("4. 클러스터별 포트폴리오 최적화 중...")

    portfolios = {}
    cluster_returns = returns.iloc[window:]  # 특성 생성에 사용된 윈도우만큼 제외

    for cluster_id in torch.unique(clusters):
        print(f"\n클러스터 {cluster_id.item()} 최적화 중...")

        # 해당 클러스터에 속하는 기간의 데이터
        mask = (clusters == cluster_id).numpy()
        cluster_data = cluster_returns[mask]

        if len(cluster_data) < 10:
            print(f"데이터 부족 (기간: {len(cluster_data)}) - 건너뜀")
            continue

        # 평균 수익률과 공분산 행렬
        mean_ret = cluster_data.mean()
        cov_matrix = cluster_data.cov()

        try:
            # 최소분산 포트폴리오 (Global Minimum Variance Portfolio)
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones((len(mean_ret), 1))
            weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
            weights = weights.flatten()

            # 음수 가중치를 0으로 설정 (공매도 금지)
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()  # 정규화

        except:
            print(f"역행렬 계산 실패 - 동일가중 포트폴리오 사용")
            weights = np.ones(len(mean_ret)) / len(mean_ret)

        portfolio_return = (weights * mean_ret).sum()
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        portfolios[cluster_id.item()] = {
            "weights": weights,
            "return": portfolio_return,
            "volatility": portfolio_vol,
            "sharpe": sharpe_ratio,
            "periods": mask.sum().item(),
        }

        print(f"최적화 완료:")
        print(f" - 일일 수익률: {portfolio_return:.4f}")
        print(f" - 일일 변동성: {portfolio_vol:.4f}")
        print(f" - 샤프 비율: {sharpe_ratio:.3f}")
        print(f" - 분석 기간: {mask.sum().item()}일")

    print(f"\n총 {len(portfolios)}개 포트폴리오 최적화 완료")
    return portfolios


# %%
def select_best_portfolio(portfolios):
    print("5. 최적 포트폴리오 선택 중...")

    if not portfolios:
        print("최적화된 포트폴리오가 없습니다!")
        return None, None

    best_cluster = min(portfolios.keys(), key=lambda k: portfolios[k]["volatility"])
    best_portfolio = portfolios[best_cluster]

    print(f"최적 포트폴리오 선택 완료:")
    print(f" - 선택된 클러스터: {best_cluster}")
    print(f" - 변동성: {best_portfolio['volatility']:.4f}")
    print(f" - 다른 클러스터 변동성:")
    for cluster_id, portfolio in portfolios.items():
        if cluster_id != best_cluster:
            print(f"    - 클러스터 {cluster_id}: {portfolio['volatility']:.4f}")

    return best_cluster, best_portfolio


# %%
def print_portfolio_results(returns, best_cluster, best_portfolio, portfolios):
    print(f"\n" + "=" * 50)
    print(f"6. 최적 포트폴리오 결과 (클러스터 {best_cluster})")
    print("=" * 50)

    # 자산별 가중치
    asset_names = returns.columns
    print(f"\n포트폴리오 구성:")
    for i, asset in enumerate(asset_names):
        print(f" - {asset}: {best_portfolio['weights'][i]:.1%}")

    # 성과 지표
    print(f"\n예상 성과 (일일 기준):")
    print(
        f" - 수익률: {best_portfolio['return']:.4f} ({best_portfolio['return']*100:.2f}%)"
    )
    print(
        f" - 변동성: {best_portfolio['volatility']:.4f} ({best_portfolio['volatility']*100:.2f}%)"
    )
    print(f" - 샤프 비율: {best_portfolio['sharpe']:.3f}")
    print(f" - 분석 기간: {best_portfolio['periods']} 거래일")

    annual_return = best_portfolio["return"] * 252
    annual_vol = best_portfolio["volatility"] * np.sqrt(252)
    annual_sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    print(f"\n연율화 성과:")
    print(f" -  연간 수익률: {annual_return:.1%}")
    print(f" -  연간 변동성: {annual_vol:.1%}")
    print(f" -  연간 샤프 비율: {annual_sharpe:.3f}")

    print(f"\n클러스터별 비교:")
    for cluster_id in sorted(portfolios.keys()):
        portfolio = portfolios[cluster_id]
        marker = " ⭐" if cluster_id == best_cluster else ""
        print(
            f" - 클러스터 {cluster_id}: "
            f"변동성 {portfolio['volatility']:.4f}, "
            f"수익률 {portfolio['return']:.4f}, "
            f"샤프 {portfolio['sharpe']:.3f}{marker}"
        )


# %%
if __name__ == "__main__":
    returns = load_data("data/stock_bond_futures_data.csv")
    features = create_features(returns, window=60)
    clusters, centers, scaler = random_walk_clustering(features, n_clusters=4)
    portfolios = optimize_portfolios(returns, clusters, window=60)
    best_cluster, best_portfolio = select_best_portfolio(portfolios)
    print_portfolio_results(returns, best_cluster, best_portfolio, portfolios)

# %%
