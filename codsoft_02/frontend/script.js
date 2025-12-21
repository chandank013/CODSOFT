// Credit Card Fraud Detection - Frontend JavaScript

// Initialize feature inputs
document.addEventListener('DOMContentLoaded', function() {
    initializeFeatureInputs();
    setupEventListeners();
});

// Create V1-V28 feature inputs
function initializeFeatureInputs() {
    const features1 = document.getElementById('features1');
    const features2 = document.getElementById('features2');
    
    // V1-V14
    for (let i = 1; i <= 14; i++) {
        features1.appendChild(createFeatureInput(i));
    }
    
    // V15-V28
    for (let i = 15; i <= 28; i++) {
        features2.appendChild(createFeatureInput(i));
    }
}

function createFeatureInput(num) {
    const div = document.createElement('div');
    div.className = 'feature-input-group';
    
    const label = document.createElement('label');
    label.textContent = `V${num}:`;
    label.setAttribute('for', `v${num}`);
    
    const input = document.createElement('input');
    input.type = 'number';
    input.id = `v${num}`;
    input.step = '0.01';
    input.value = '0.00';
    input.placeholder = '0.00';
    
    div.appendChild(label);
    div.appendChild(input);
    
    return div;
}

// Setup event listeners
function setupEventListeners() {
    // Toggle advanced features
    document.getElementById('toggleFeatures').addEventListener('click', function() {
        const features = document.getElementById('advancedFeatures');
        const icon = document.getElementById('toggleIcon');
        
        features.classList.toggle('hidden');
        icon.classList.toggle('open');
    });
    
    // Randomize features button
    document.getElementById('randomizeFeatures').addEventListener('click', randomizeFeatures);
    
    // Analyze button
    document.getElementById('analyzeBtn').addEventListener('click', analyzeTransaction);
    
    // Toggle transaction details
    document.getElementById('toggleDetails').addEventListener('click', function() {
        const details = document.getElementById('transactionDetails');
        const icon = document.getElementById('detailsIcon');
        
        details.classList.toggle('hidden');
        icon.classList.toggle('open');
    });
}

// Randomize V features
function randomizeFeatures() {
    for (let i = 1; i <= 28; i++) {
        const input = document.getElementById(`v${i}`);
        // Random value between -3 and 3 (typical range for PCA features)
        input.value = (Math.random() * 6 - 3).toFixed(2);
    }
}

// Load sample transaction
function loadSample(type) {
    const samples = {
        legitimate: {
            id: 'TXN_LEGIT_001',
            amount: 45.99,
            time: 5000,
            features: Array(28).fill(0).map(() => (Math.random() * 0.4 - 0.2).toFixed(2))
        },
        suspicious: {
            id: 'TXN_SUSP_002',
            amount: 125.00,
            time: 8000,
            features: Array(28).fill(0).map(() => (Math.random() * 1.6 - 0.8).toFixed(2))
        },
        fraud: {
            id: 'TXN_FRAUD_003',
            amount: 250.50,
            time: 12000,
            features: [
                -2.5, 3.1, -1.8, 2.9, -0.8, 1.2, -1.5, 0.9, -2.1, 2.3,
                -1.7, 1.9, -0.7, -3.2, 1.1, -2.8, 2.5, 0.6, -1.3, 0.8,
                -0.9, 1.4, -1.1, 0.5, -0.6, 0.7, -0.4, 0.3
            ]
        }
    };
    
    const sample = samples[type];
    
    // Load basic fields
    document.getElementById('transactionId').value = sample.id;
    document.getElementById('amount').value = sample.amount;
    document.getElementById('time').value = sample.time;
    
    // Load V features
    for (let i = 0; i < 28; i++) {
        document.getElementById(`v${i + 1}`).value = sample.features[i];
    }
    
    // Scroll to form
    document.querySelector('.input-card').scrollIntoView({ behavior: 'smooth' });
}

// Analyze transaction
async function analyzeTransaction() {
    // Get form data
    const transactionId = document.getElementById('transactionId').value;
    const amount = parseFloat(document.getElementById('amount').value);
    const time = parseFloat(document.getElementById('time').value);
    
    // Validate
    if (!transactionId || isNaN(amount) || isNaN(time)) {
        alert('Please fill in all required fields');
        return;
    }
    
    // Collect V features
    const vFeatures = {};
    for (let i = 1; i <= 28; i++) {
        vFeatures[`V${i}`] = parseFloat(document.getElementById(`v${i}`).value) || 0;
    }
    
    // Show loading
    document.getElementById('results').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');
    
    // Simulate API call (in production, this would call your Flask/FastAPI backend)
    setTimeout(() => {
        const result = simulateFraudDetection(amount, time, vFeatures);
        displayResults(result, transactionId, amount);
    }, 1500);
}

// Simulate fraud detection (mock prediction)
function simulateFraudDetection(amount, time, vFeatures) {
    // Simple heuristic for demo purposes
    // In production, this would call your actual ML model via API
    
    let riskScore = 0;
    
    // Check amount
    if (amount > 200) riskScore += 0.2;
    else if (amount > 100) riskScore += 0.1;
    
    // Check V features (simplified - checking for extreme values)
    let extremeFeatures = 0;
    for (let key in vFeatures) {
        const value = Math.abs(vFeatures[key]);
        if (value > 2) extremeFeatures++;
        if (value > 3) extremeFeatures += 2;
    }
    
    riskScore += (extremeFeatures / 28) * 0.6;
    
    // Add some randomness for variety
    riskScore += Math.random() * 0.2;
    
    // Cap at 1.0
    riskScore = Math.min(riskScore, 1.0);
    
    // Determine if fraud
    const isFraud = riskScore > 0.5;
    
    // Determine risk level
    let riskLevel;
    if (riskScore >= 0.8) riskLevel = 'CRITICAL';
    else if (riskScore >= 0.6) riskLevel = 'HIGH';
    else if (riskScore >= 0.4) riskLevel = 'MEDIUM';
    else if (riskScore >= 0.2) riskLevel = 'LOW';
    else riskLevel = 'MINIMAL';
    
    // Generate recommendation
    let recommendation;
    if (isFraud) {
        if (riskScore >= 0.8) {
            recommendation = '🚨 BLOCK: Immediate action required - Very high fraud risk. Block this transaction and contact the cardholder immediately.';
        } else if (riskScore >= 0.6) {
            recommendation = '⚠️ REVIEW: Manual review recommended - High fraud risk. Flag this transaction for detailed investigation.';
        } else {
            recommendation = '👁️ MONITOR: Flag for monitoring - Moderate fraud risk. Watch for similar patterns from this account.';
        }
    } else {
        recommendation = '✅ APPROVE: Transaction appears legitimate based on current analysis. Normal processing can proceed.';
    }
    
    return {
        isFraud,
        probability: riskScore,
        riskLevel,
        recommendation,
        timestamp: new Date().toISOString()
    };
}

// Display results
function displayResults(result, transactionId, amount) {
    // Hide loading
    document.getElementById('loading').classList.add('hidden');
    
    // Show results
    document.getElementById('results').classList.remove('hidden');
    
    // Status badge
    const statusBadge = document.getElementById('statusBadge');
    if (result.isFraud) {
        statusBadge.textContent = '🚨 FRAUDULENT TRANSACTION DETECTED';
        statusBadge.className = 'status-badge fraud';
    } else {
        statusBadge.textContent = '✅ LEGITIMATE TRANSACTION';
        statusBadge.className = 'status-badge legitimate';
    }
    
    // Update metrics
    document.getElementById('probability').textContent = `${(result.probability * 100).toFixed(1)}%`;
    document.getElementById('riskLevel').textContent = result.riskLevel;
    document.getElementById('displayAmount').textContent = `$${amount.toFixed(2)}`;
    document.getElementById('displayId').textContent = transactionId;
    
    // Update probability bar
    const probabilityFill = document.getElementById('probabilityFill');
    probabilityFill.style.width = `${result.probability * 100}%`;
    
    // Update recommendation
    const recommendationBox = document.getElementById('recommendation');
    recommendationBox.textContent = result.recommendation;
    
    if (result.isFraud) {
        if (result.probability >= 0.8) {
            recommendationBox.className = 'recommendation-box block';
        } else {
            recommendationBox.className = 'recommendation-box review';
        }
    } else {
        recommendationBox.className = 'recommendation-box approve';
    }
    
    // Create transaction details
    createTransactionDetails(transactionId, amount, result);
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create transaction details display
function createTransactionDetails(transactionId, amount, result) {
    const detailsDiv = document.getElementById('transactionDetails');
    detailsDiv.innerHTML = '';
    
    const details = [
        { label: 'Transaction ID', value: transactionId },
        { label: 'Amount', value: `$${amount.toFixed(2)}` },
        { label: 'Fraud Probability', value: `${(result.probability * 100).toFixed(2)}%` },
        { label: 'Risk Level', value: result.riskLevel },
        { label: 'Classification', value: result.isFraud ? 'FRAUD' : 'LEGITIMATE' },
        { label: 'Analysis Time', value: new Date(result.timestamp).toLocaleString() },
        { label: 'Model Version', value: 'v1.0 (Random Forest + SMOTE)' },
        { label: 'Confidence', value: result.probability > 0.7 || result.probability < 0.3 ? 'HIGH' : 'MEDIUM' }
    ];
    
    details.forEach(detail => {
        const row = document.createElement('div');
        row.className = 'detail-row';
        
        const label = document.createElement('span');
        label.className = 'detail-label';
        label.textContent = detail.label + ':';
        
        const value = document.createElement('span');
        value.className = 'detail-value';
        value.textContent = detail.value;
        
        row.appendChild(label);
        row.appendChild(value);
        detailsDiv.appendChild(row);
    });
}

// Make loadSample globally accessible
window.loadSample = loadSample;

// Note for production:
// In a real implementation, you would:
// 1. Have a Flask/FastAPI backend serving your trained model
// 2. Replace simulateFraudDetection() with an actual API call:
/*
async function analyzeTransaction() {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            transactionId,
            amount,
            time,
            ...vFeatures
        })
    });
    
    const result = await response.json();
    displayResults(result, transactionId, amount);
}
*/