import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

def add_features(df):
    df['current_run_rate'] = df.apply(lambda row: row['runs'] / row['over'] if row['over'] > 0 else 0, axis=1)
    df['remaining_overs'] = 20 - df['over']
    df['wickets_in_hand'] = 10 - df['wickets']
    df['is_death_overs'] = (df['over'] >= 16).astype(int)
    df['projected_score'] = df['current_run_rate'] * 20
    df['pressure_factor'] = df['wickets_in_hand'] * df['current_run_rate']
    return df

def main():
    print("✅ Loading dataset...")
    df = pd.read_csv('data/ipl.csv')  # assume already refined & has mid, over etc.

    print("⚙ Adding cricket features...")
    df = add_features(df)

    # Define features & target
    features = ['over', 'runs', 'wickets', 'venue', 'bat_team', 'bowl_team',
            'current_run_rate', 'remaining_overs', 'wickets_in_hand', 'is_death_overs',
            'projected_score', 'pressure_factor']

    target = 'total'

    X = df[features]
    y = df[target]

    numeric_features = ['over', 'runs', 'wickets', 'current_run_rate', 'remaining_overs',
                    'wickets_in_hand', 'is_death_overs', 'projected_score', 'pressure_factor']

    categorical_features = ['venue', 'bat_team', 'bowl_team']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    print("⚙ Transforming data...")
    X_transformed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.1, random_state=42)

    print("🔍 Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"📊 Model Performance: R²: {r2:.2f}, MAE: {mae:.2f}")

    # Save model & preprocessor
    joblib.dump(model, 'data/best_model.pkl')
    joblib.dump(preprocessor, 'data/preprocessor.pkl')
    print("🎉 Done! Saved best_model.pkl & preprocessor.pkl.")

if __name__ == "__main__":
    main()
