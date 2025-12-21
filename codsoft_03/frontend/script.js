// Customer Churn Prediction - Frontend JavaScript

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    document.getElementById('predictBtn').addEventListener('click', predictChurn);
}

// Load sample customer
function loadSample(type) {
    const samples = {
        loyal: {
            creditScore: 750,
            geography: 'France',
            gender: 'Female',
            age: 35,
            tenure: 8,
            balance: 125000,
            numProducts: 2,
            hasCrCard: 1,
            isActive: 1,
            salary: 80000
        },
        atrisk: {
            creditScore: 600,
            geography: 'Spain',
            gender: 'Female',
            age: 42,
            tenure: 5,
            balance: 75000,
            numProducts: 1,
            hasCrCard: 1,
            isActive: 0,
            salary: 60000
        },
        churning: {
            creditScore: 400,
            geography: 'Germany',
            gender: 'Male',
            age: 55,
            tenure: 2,
            balance: 0,
            numProducts: 1,
            hasCrCard: 0,
            isActive: 0,
            salary: 45000
        }
    };
    
    const sample = samples[type];
    
    // Load values
    document.getElementById('creditScore').value = sample.creditScore;
    document.getElementById('geography').value = sample.geography;
    document.getElementById('gender').value = sample.gender;
    document.getElementById('age').value = sample.age;
    document.getElementById('tenure').value = sample.tenure;
    document.getElementById('balance').value = sample.balance;
    document.getElementById('numProducts').value = sample.numProducts;
    document.getElementById('hasCrCard').value = sample.hasCrCard;
    document.getElementById('isActive').value = sample.isActive;
    document.getElementById('salary').value = sample.salary;
    
    // Scroll to form
    document.querySelector('.input-card').scrollIntoView({ behavior: 'smooth' });
}

// Predict churn
async function predictChurn() {
    // Get form data
    const customerData = {
        creditScore: parseInt(document.getElementById('creditScore').value),
        geography: document.getElementById('geography').value,
        gender: document.getElementById('gender').value,
        age: parseInt(document.getElementById('age').value),
        tenure: parseInt(document.getElementById('tenure').value),
        balance: parseFloat(document.getElementById('balance').value),
        numProducts: parseInt(document.getElementById('numProducts').value),
        hasCrCard: parseInt(document.getElementById('hasCrCard').value),
        isActive: parseInt(document.getElementById('isActive').value),
        salary: parseFloat(document.getElementById('salary').value)
    };
    
    // Validate
    if (!validateInput(customerData)) {
        alert('Please fill in all fields with valid values');
        return;
    }
    
    // Show loading
    document.getElementById('results').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');
    
    // Simulate API call
    setTimeout(() => {
        const result = simulateChurnPrediction(customerData);
        displayResults(result, customerData);
    }, 1500);
}

// Validate input
function validateInput(data) {
    return data.creditScore >= 300 && data.creditScore <= 850 &&
           data.age >= 18 && data.age <= 100 &&
           data.tenure >= 0 && data.tenure <= 10 &&
           data.balance >= 0 &&
           data.numProducts >= 1 && data.numProducts <= 4 &&
           data.salary >= 10000;
}

// Simulate churn prediction
function simulateChurnPrediction(customer) {
    // Simple heuristic for demo
    let churnScore = 0;
    
    // Credit score impact (inverse)
    churnScore += (850 - customer.creditScore) / 850 * 0.15;
    
    // Age impact
    if (customer.age > 50) churnScore += 0.15;
    else if (customer.age < 30) churnScore += 0.05;
    
    // Tenure impact (inverse)
    churnScore += (10 - customer.tenure) / 10 * 0.20;
    
    // Balance impact
    if (customer.balance === 0) churnScore += 0.25;
    else if (customer.balance < 50000) churnScore += 0.10;
    
    // Products impact
    if (customer.numProducts === 1) churnScore += 0.15;
    else if (customer.numProducts > 2) churnScore += 0.10;
    
    // Activity impact
    if (customer.isActive === 0) churnScore += 0.20;
    
    // Credit card impact
    if (customer.hasCrCard === 0) churnScore += 0.05;
    
    // Geography impact
    if (customer.geography === 'Germany') churnScore += 0.10;
    else if (customer.geography === 'Spain') churnScore += 0.05;
    
    // Gender impact
    if (customer.gender === 'Female') churnScore += 0.05;
    
    // Cap at 1.0
    churnScore = Math.min(churnScore, 0.95);
    churnScore = Math.max(churnScore, 0.05);
    
    // Determine if churn
    const willChurn = churnScore > 0.5;
    
    // Risk level
    let riskLevel, riskColor;
    if (churnScore >= 0.8) {
        riskLevel = 'CRITICAL';
        riskColor = '🔴';
    } else if (churnScore >= 0.6) {
        riskLevel = 'HIGH';
        riskColor = '🟠';
    } else if (churnScore >= 0.4) {
        riskLevel = 'MEDIUM';
        riskColor = '🟡';
    } else if (churnScore >= 0.2) {
        riskLevel = 'LOW';
        riskColor = '🟢';
    } else {
        riskLevel = 'MINIMAL';
        riskColor = '🟢';
    }
    
    // Recommendation
    let recommendation;
    if (willChurn) {
        if (churnScore >= 0.8) {
            recommendation = '🚨 IMMEDIATE ACTION: Very high churn risk. Contact customer immediately with retention offer.';
        } else if (churnScore >= 0.6) {
            recommendation = '⚠️ HIGH PRIORITY: Significant churn risk. Schedule personalized outreach within 48 hours.';
        } else {
            recommendation = '📊 MONITOR: Moderate churn risk. Include in retention campaign and monitor activity.';
        }
    } else {
        recommendation = '✅ RETAIN: Customer likely to stay. Continue regular engagement and quality service.';
    }
    
    // Retention strategies
    const strategies = generateRetentionStrategies(customer, churnScore);
    
    return {
        willChurn,
        churnProbability: churnScore,
        riskLevel,
        riskColor,
        recommendation,
        strategies
    };
}

// Generate retention strategies
function generateRetentionStrategies(customer, probability) {
    const strategies = [];
    
    if (customer.isActive === 0) {
        strategies.push({
            icon: '✨',
            text: 'Activate engagement: Send personalized re-engagement campaign with exclusive offers'
        });
    }
    
    if (customer.numProducts === 1) {
        strategies.push({
            icon: '📦',
            text: 'Cross-sell opportunity: Offer additional products with 20% discount for bundle'
        });
    }
    
    if (customer.age > 50) {
        strategies.push({
            icon: '👴',
            text: 'Senior care program: Provide dedicated support line and personalized service'
        });
    }
    
    if (customer.balance === 0) {
        strategies.push({
            icon: '💰',
            text: 'Incentivize deposits: Offer 2% interest bonus on new deposits over $10,000'
        });
    }
    
    if (customer.hasCrCard === 0) {
        strategies.push({
            icon: '💳',
            text: 'Credit card promotion: Offer premium card with cashback rewards and no annual fee'
        });
    }
    
    if (customer.tenure < 3) {
        strategies.push({
            icon: '🎁',
            text: 'Loyalty program: Enroll in rewards program with immediate sign-up bonus'
        });
    }
    
    if (probability > 0.6) {
        strategies.push({
            icon: '👨‍💼',
            text: 'VIP treatment: Assign dedicated relationship manager for personalized service'
        });
        strategies.push({
            icon: '🎯',
            text: 'Retention offer: Provide exclusive benefits package worth $500 annually'
        });
    }
    
    if (strategies.length === 0) {
        strategies.push({
            icon: '✅',
            text: 'Continue standard engagement and maintain high service quality'
        });
    }
    
    return strategies;
}

// Display results
function displayResults(result, customerData) {
    // Hide loading
    document.getElementById('loading').classList.add('hidden');
    
    // Show results
    document.getElementById('results').classList.remove('hidden');
    
    // Status badge
    const statusBadge = document.getElementById('statusBadge');
    if (result.willChurn) {
        statusBadge.textContent = '🚨 CUSTOMER LIKELY TO CHURN';
        statusBadge.className = 'status-badge will-churn';
    } else {
        statusBadge.textContent = '✅ CUSTOMER LIKELY TO STAY';
        statusBadge.className = 'status-badge will-stay';
    }
    
    // Update metrics
    document.getElementById('probability').textContent = `${(result.churnProbability * 100).toFixed(1)}%`;
    document.getElementById('riskLevel').textContent = `${result.riskColor} ${result.riskLevel}`;
    document.getElementById('displayProducts').textContent = customerData.numProducts;
    document.getElementById('displayTenure').textContent = `${customerData.tenure} years`;
    
    // Update probability bar
    const probabilityFill = document.getElementById('probabilityFill');
    probabilityFill.style.width = `${result.churnProbability * 100}%`;
    
    // Update recommendation
    const recommendationBox = document.getElementById('recommendation');
    recommendationBox.textContent = result.recommendation;
    
    if (result.willChurn) {
        if (result.churnProbability >= 0.8) {
            recommendationBox.className = 'recommendation-box action';
        } else if (result.churnProbability >= 0.6) {
            recommendationBox.className = 'recommendation-box action';
        } else {
            recommendationBox.className = 'recommendation-box monitor';
        }
    } else {
        recommendationBox.className = 'recommendation-box retain';
    }
    
    // Display retention strategies
    const strategiesDiv = document.getElementById('strategies');
    strategiesDiv.innerHTML = '';
    
    result.strategies.forEach(strategy => {
        const strategyItem = document.createElement('div');
        strategyItem.className = 'strategy-item';
        strategyItem.innerHTML = `
            <span class="strategy-icon">${strategy.icon}</span>
            <span>${strategy.text}</span>
        `;
        strategiesDiv.appendChild(strategyItem);
    });
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Make loadSample globally accessible
window.loadSample = loadSample;

// Production note:
// In production, replace simulateChurnPrediction() with actual API call:
/*
async function predictChurn() {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(customerData)
    });
    
    const result = await response.json();
    displayResults(result, customerData);
}
*/