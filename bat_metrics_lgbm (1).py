import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("swing+ data.csv")

# Define features and target
features = ['avg_swing_speed', 'avg_swing_length', 'attack_angle', 
            'attack_direction', 'vertical_swing_path']
target = 'xwobacon'

X = data[features]
y = data[target]
weights = data['pa']

# Split data with weights
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Train LightGBM model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.1,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
valid_data = lgb.Dataset(X_test, label=y_test, weight=weights_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# Make predictions on full dataset
predictions = model.predict(X)

# Create swing+ stat (scaled to 100 = average, like stuff+)
# Calculate weighted average using PA as weights
weighted_avg_xwobacon = np.average(predictions, weights=data['pa'])

# Scale to 100 = average
swing_plus = (predictions / weighted_avg_xwobacon) * 100

# Create output dataframe with original data + predictions + swing+
output_data = data.copy()
output_data['predicted_xwobacon'] = predictions
output_data['swing_plus'] = swing_plus

# Save to CSV
output_data.to_csv('swing_predictions.csv', index=False)

# Print model performance
y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print(f"Model Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Training used PA weights - higher PA players had more influence")
print(f"\nSwing+ Stats:")
print(f"Average swing+ (weighted by PA): {np.average(swing_plus, weights=data['pa']):.1f}")
print(f"Swing+ range: {swing_plus.min():.1f} to {swing_plus.max():.1f}")
print(f"Players with swing+ > 110: {(swing_plus > 110).sum()}")
print(f"Players with swing+ < 90: {(swing_plus < 90).sum()}")

# Create feature importance chart
importance = model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title('Feature Importance for xwOBACON Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nFeature Importance:")
for i, row in importance_df.sort_values('importance', ascending=False).iterrows():
    print(f"{row['feature']}: {row['importance']:.0f}")

print(f"\nOutput saved to: swing_predictions.csv")
print(f"Chart saved to: feature_importance.png")
print(f"\nColumns in output CSV:")
print(f"- All original data")
print(f"- predicted_xwobacon: Model's xwOBACON prediction")  
print(f"- swing_plus: Scaled metric (100 = average, like Stuff+)")
