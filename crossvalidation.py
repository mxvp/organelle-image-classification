from sklearn.model_selection import KFold


# Early stopping
best_val_loss = float('inf') # Placeholder
patience = 5  # Define patience - number of epochs without improvement to tolerate
counter = 0  # Counter to keep track of epochs without improvement

num_epochs = 20
num_folds = 4  # Number of folds for cross-validation

# Initialize lists to store metrics and labels for later visualizations
train_losses_cv = []
val_losses_cv = []
train_f1_scores_cv = []
val_f1_scores_cv = []
lr_values_cv = []

# Initialize the KFold splitter
kf = KFold(n_splits=num_folds)

# Perform cross-validation on the training part
for fold, (train_index, val_index) in enumerate(kf.split(train_part_df)):
    # Create DataFrames for this fold's training and validation data
    train_fold_df = train_part_df.iloc[train_index]
    val_fold_df = train_part_df.iloc[val_index]

    # Create DataLoaders for this fold's training and validation data
    train_loader = DataLoader(ProteinDataset(train_fold_df, transform=transforms_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(ProteinDataset(val_fold_df, transform=transforms_train), batch_size=32, shuffle=False)

    
    # Initialize the model for the current fold
    model = ProteinDenseNet().to(device)
    
    # Define the criterion, optimizer, and learning rate scheduler for the current fold
    criterion = nn.BCELoss(weight=class_weights_tensor).to(device) # Criterion moved to device, Binary Cross Entropy Loss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adaptive Moment Estimation optimizer, learning rate 0.001
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=1, verbose=True)
    
    # Initialize variables for metrics and early stopping for the current fold
    best_val_loss = float('inf')
    counter = 0
    
    # Iterate through each epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_f1 = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            f1_batch = calculate_f1_score(outputs, labels)
            running_f1 += f1_batch

        avg_train_loss = running_loss / len(train_loader)
        avg_train_f1 = running_f1 / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_f1 = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                f1_batch = calculate_f1_score(outputs, labels)
                val_f1 += f1_batch

        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)

        print(f"Fold {fold+1}/{num_folds}, Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val F1: {avg_val_f1:.4f}")

        train_losses_cv.append(avg_train_loss)
        val_losses_cv.append(avg_val_loss)
        train_f1_scores_cv.append(avg_train_f1)
        val_f1_scores_cv.append(avg_val_f1)
        lr_values_cv.append(optimizer.param_groups[0]['lr'])

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping: No improvement for {patience} consecutive epochs.')
                break

## Visualisations
            
epochs = range(1, 52)

# Plotting loss and F1-score in subplots
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Plotting Loss
axs[0].plot(epochs, train_losses_cv, label='Train Loss')
axs[0].plot(epochs, val_losses_cv, label='Validation Loss')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plotting F1-score
axs[1].plot(epochs, train_f1_scores_cv, label='Train F1-score')
axs[1].plot(epochs, val_f1_scores_cv, label='Validation F1-score')
axs[1].set_title('Training and Validation F1-score')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('F1-score')
axs[1].legend()

plt.tight_layout()
plt.show()


# Plotting learning rate changes over epochs
plt.figure(figsize=(6, 4))
plt.plot(range(len(lr_values_cv)), lr_values_cv)
plt.title('Learning Rate Schedule')
plt.xlabel('Iterations/Epochs')
plt.ylabel('Learning Rate')
plt.show()