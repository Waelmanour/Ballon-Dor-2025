import streamlit as st
import numpy as np
import altair as alt
import pandas as pd

st.set_page_config(page_title="Ballon d'Or Predictor", page_icon="🏆", layout="centered")
st.title("wael predict the winner")

# -------------------------------
# Synthetic training data (numpy-based)
# -------------------------------

def generate_synthetic_seasons(num_seasons: int = 40, players_per_season: int = 22, random_state: int = 7):
    rng = np.random.default_rng(random_state)
    X_list = []
    y_list = []
    for _ in range(num_seasons):
        goals = rng.poisson(lam=rng.uniform(8, 30), size=players_per_season)
        assists = rng.poisson(lam=rng.uniform(5, 15), size=players_per_season)
        minutes = rng.integers(1500, 4800, size=players_per_season)
        rating = np.clip(rng.normal(7.2, 0.35, size=players_per_season), 6.2, 8.6)
        major_trophies = rng.binomial(2, 0.2, size=players_per_season)
        base_score = (
            0.035 * goals +
            0.025 * assists +
            0.0002 * minutes +
            1.20 * (rating - 7.0) +
            0.85 * major_trophies +
            rng.normal(0, 0.1, size=players_per_season)
        )
        winner_idx = int(np.argmax(base_score))
        season_X = np.column_stack([goals, assists, minutes, rating, major_trophies])
        season_y = np.zeros(players_per_season, dtype=int)
        season_y[winner_idx] = 1
        X_list.append(season_X)
        y_list.append(season_y)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

FEATURES = ["goals", "assists", "minutes", "rating", "major_trophies"]

# -------------------------------
# Custom logistic regression (pure NumPy)
# -------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class NumpyLogReg:
    def __init__(self, lr=0.01, epochs=2000, l2=0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    def fit(self, X, y):
        # standardize
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sigma
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z = Xs @ self.w + self.b
            p = sigmoid(z)
            # gradients
            grad_w = (Xs.T @ (p - y)) / n + self.l2 * self.w
            grad_b = np.mean(p - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = (X - self.mu) / self.sigma
        z = Xs @ self.w + self.b
        p = sigmoid(z)
        return np.column_stack([1 - p, p])

# Train model
@st.cache_resource
def train_model():
    X, y = generate_synthetic_seasons()
    # address imbalance by upweighting positives in training loop
    # simple trick: duplicate winners
    X_aug = np.vstack([X, X[y == 1]])
    y_aug = np.concatenate([y, np.ones_like(y[y == 1])])
    model = NumpyLogReg(lr=0.03, epochs=2500, l2=0.001).fit(X_aug, y_aug)
    # simple CV proxy: split into 5 chunks and compute AUC approx via rank correlation
    chunks = np.array_split(np.arange(len(X)), 5)
    aucs = []
    for idx in chunks:
        Xc, yc = X[idx], y[idx]
        probs = model.predict_proba(Xc)[:, 1]
        # AUC approximation using rank concordance
        pos = probs[yc == 1]
        neg = probs[yc == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # fraction of pairs where pos>neg
        pairs = 0
        wins = 0
        for pp in pos:
            wins += np.sum(pp > neg)
            pairs += len(neg)
        aucs.append(wins / max(pairs, 1))
    return model, np.array(aucs) if len(aucs) else np.array([0.5])

model, auc_scores = train_model()

st.subheader("Model Overview")
st.write("This app trains a professional logistic-style model on a domain-inspired synthetic dataset to estimate each candidate's probability of winning the Ballon d'Or based on performance and trophies.")

st.subheader("Ballon d'Or Award Criteria")
st.write("The Ballon d'Or is awarded based on three main criteria:")
criteria = [
    "**1. Individual Performance**: Decisive and impressive character",
    "**2. Team Performance**: Achievements and trophies",
    "**3. Class and Fair Play**: Sportsmanship and conduct"
]
for criterion in criteria:
    st.markdown(criterion)

# -------------------------------
# Candidate inputs
# -------------------------------

def default_candidates():
    return [
        {"player": "Ousmane Dembélé (PSG)", "goals": 35, "assists": 14, "minutes": 4100, "rating": 8.5, "major_trophies": 2},
        {"player": "Jude Bellingham (Real Madrid)", "goals": 15, "assists": 14, "minutes": 3800, "rating": 8.3, "major_trophies": 1},
        {"player": "Raphinha (FC Barcelona)", "goals": 13, "assists": 10, "minutes": 3600, "rating": 8.2, "major_trophies": 1},
        {"player": "Kylian Mbappé (Real Madrid)", "goals": 31, "assists": 8, "minutes": 3900, "rating": 8.1, "major_trophies": 1},
        {"player": "Lamine Yamal (FC Barcelona)", "goals": 16, "assists": 23, "minutes": 3700, "rating": 8.0, "major_trophies": 1},
        {"player": "Vinícius Jr. (Real Madrid)", "goals": 18, "assists": 12, "minutes": 3650, "rating": 7.9, "major_trophies": 1},
        {"player": "Erling Haaland (Manchester City)", "goals": 34, "assists": 5, "minutes": 3800, "rating": 7.9, "major_trophies": 1},
        {"player": "Serhou Guirassy (Borussia Dortmund)", "goals": 13, "assists": 3, "minutes": 2800, "rating": 7.8, "major_trophies": 0},
        {"player": "Viktor Gyökeres (Sporting CP)", "goals": 54, "assists": 8, "minutes": 4200, "rating": 7.8, "major_trophies": 1},
        {"player": "Robert Lewandowski (FC Barcelona)", "goals": 24, "assists": 6, "minutes": 3600, "rating": 7.8, "major_trophies": 1},
        {"player": "Harry Kane (Bayern Munich)", "goals": 26, "assists": 7, "minutes": 3700, "rating": 7.7, "major_trophies": 1},
        {"player": "Achraf Hakimi (PSG)", "goals": 8, "assists": 15, "minutes": 3800, "rating": 7.7, "major_trophies": 2},
        {"player": "Désiré Doué (PSG)", "goals": 16, "assists": 14, "minutes": 3400, "rating": 7.7, "major_trophies": 2},
        {"player": "Khvicha Kvaratskhelia (PSG/Napoli)", "goals": 14, "assists": 10, "minutes": 3600, "rating": 7.7, "major_trophies": 2},
        {"player": "Lautaro Martínez (Inter Milan)", "goals": 18, "assists": 6, "minutes": 3500, "rating": 7.6, "major_trophies": 1},
        {"player": "Scott McTominay (Napoli)", "goals": 12, "assists": 5, "minutes": 3300, "rating": 7.6, "major_trophies": 1},
        {"player": "Denzel Dumfries (Inter Milan)", "goals": 11, "assists": 7, "minutes": 3400, "rating": 7.6, "major_trophies": 1},
        {"player": "Alexis Mac Allister (Liverpool)", "goals": 7, "assists": 10, "minutes": 3600, "rating": 7.5, "major_trophies": 0},
        {"player": "Nuno Mendes (PSG)", "goals": 3, "assists": 8, "minutes": 3400, "rating": 7.5, "major_trophies": 2},
        {"player": "Gianluigi Donnarumma (PSG)", "goals": 0, "assists": 0, "minutes": 3800, "rating": 7.5, "major_trophies": 2},
        {"player": "Cole Palmer (Chelsea)", "goals": 20, "assists": 12, "minutes": 3700, "rating": 7.5, "major_trophies": 0},
        {"player": "Rodri (Manchester City)", "goals": 5, "assists": 7, "minutes": 2800, "rating": 7.5, "major_trophies": 1},
        {"player": "Florian Wirtz (Bayer Leverkusen)", "goals": 15, "assists": 18, "minutes": 3500, "rating": 7.5, "major_trophies": 1},
        {"player": "Mohamed Salah (Liverpool)", "goals": 22, "assists": 11, "minutes": 3600, "rating": 7.4, "major_trophies": 0},
        {"player": "Declan Rice (Arsenal)", "goals": 7, "assists": 9, "minutes": 3700, "rating": 7.4, "major_trophies": 0},
        {"player": "João Neves (PSG)", "goals": 5, "assists": 7, "minutes": 3300, "rating": 7.4, "major_trophies": 2},
        {"player": "Virgil van Dijk (Liverpool)", "goals": 3, "assists": 1, "minutes": 3800, "rating": 7.4, "major_trophies": 0},
        {"player": "Pedri (FC Barcelona)", "goals": 6, "assists": 9, "minutes": 3100, "rating": 7.3, "major_trophies": 1},
        {"player": "Fabián Ruiz (PSG)", "goals": 7, "assists": 9, "minutes": 3400, "rating": 7.3, "major_trophies": 2},
        {"player": "Michael Olise (Bayern Munich)", "goals": 11, "assists": 8, "minutes": 3000, "rating": 7.3, "major_trophies": 0},
    ]

st.subheader("Ballon d'Or 2025 Nominees")
st.caption("Tip: Adjust numbers to reflect end-of-season performance. Major trophies is a count of top-tier honors (e.g., UCL, World Cup, continental).")

# Create tabs for better organization
tab1, tab2 = st.tabs(["All Nominees", "Top Contenders"])

with tab1:
    edited = st.data_editor(
        default_candidates(),
        num_rows="dynamic",
        column_config={
            "player": st.column_config.TextColumn("Player", required=True),
            "goals": st.column_config.NumberColumn("Goals", min_value=0, step=1),
            "assists": st.column_config.NumberColumn("Assists", min_value=0, step=1),
            "minutes": st.column_config.NumberColumn("Minutes", min_value=0, step=10),
            "rating": st.column_config.NumberColumn("Average rating", min_value=5.5, max_value=9.5, step=0.05),
            "major_trophies": st.column_config.NumberColumn("Major trophies", min_value=0, max_value=3, step=1),
        },
        key="candidates_editor",
        height=400
    )
    
with tab2:
    st.write("Top contenders based on current stats:")
    top_candidates = sorted(default_candidates(), key=lambda x: (x["rating"]*2 + x["goals"]*0.1 + x["assists"]*0.05 + x["major_trophies"]), reverse=True)[:5]
    for i, player in enumerate(top_candidates):
        st.write(f"{i+1}. **{player['player']}** - {player['goals']} goals, {player['assists']} assists, {player['rating']} rating, {player['major_trophies']} trophies")

# -------------------------------
# Prediction
# -------------------------------

predict_btn = st.button("Predict Winner", type="primary")
if predict_btn:
    candidates = edited if isinstance(edited, list) else edited.to_dict(orient="records")
    if len(candidates) == 0:
        st.warning("Please add at least one candidate.")
    else:
        Xc = np.array([[c["goals"], c["assists"], c["minutes"], c["rating"], c["major_trophies"]] for c in candidates])
        probs = model.predict_proba(Xc)[:, 1]
        for i, p in enumerate(probs):
            candidates[i]["win_probability"] = float(p)
        candidates.sort(key=lambda x: x["win_probability"], reverse=True)

        st.success("Prediction complete!")
        
        # Show winner with prominence
        winner = candidates[0]
        st.markdown(f"## 🏆 Predicted Winner: {winner['player']}")
        st.markdown(f"### Probability: {winner['win_probability']:.2%}")
        
        # Create tabs for results visualization
        result_tab1, result_tab2 = st.tabs(["Top 5 Contenders", "All Nominees"])
        
        with result_tab1:
            # Show top 5 with detailed stats
            st.subheader("Top 5 Contenders")
            top5 = candidates[:5]
            
            # Bar chart for top 5
            top5_chart = alt.Chart(pd.DataFrame(top5)).mark_bar(color="#FFA500").encode(
                x=alt.X("win_probability:Q", title="Win Probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("player:N", sort="-x", title=""),
                tooltip=["player", alt.Tooltip("win_probability", format=".2%"), 
                         "goals", "assists", "rating", "major_trophies"]
            ).properties(height=200)
            
            st.altair_chart(top5_chart, use_container_width=True)
            
            # Display stats for top 5
            for i, player in enumerate(top5):
                st.write(f"{i+1}. **{player['player']}** - {player['win_probability']:.2%} probability")
                st.write(f"   Stats: {player['goals']} goals, {player['assists']} assists, {player['rating']} rating, {player['major_trophies']} trophies")
        
        with result_tab2:
            # Show all nominees in a scrollable dataframe
            st.subheader("All Nominees Ranking")
            st.dataframe(
                pd.DataFrame(candidates).rename(columns={"win_probability": "Probability"}),
                column_config={"Probability": st.column_config.ProgressColumn("Win Probability", format="%.2f", min_value=0, max_value=1)},
                height=400
            )
            
            # Bar chart for all nominees
            all_chart = alt.Chart(pd.DataFrame(candidates)).mark_bar().encode(
                x=alt.X("win_probability:Q", title="Win Probability"),
                y=alt.Y("player:N", sort="-x", title=""),
                color=alt.Color("win_probability:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["player", alt.Tooltip("win_probability", format=".2%")]
            ).properties(height=600)
            st.altair_chart(all_chart, use_container_width=True)

        winner = candidates[0]
        st.header(f"🏆 Predicted Winner: {winner['player']} ({winner['win_probability']:.2%})")

st.divider()
st.caption("This tool is for decision support and demonstration. Actual award results depend on voters and context.")