// Sample messages
const sampleMessages = [
    "WINNER!! You have won a £1000 prize. Call 09061701461 now!",
    "Hey! Are we still meeting for lunch tomorrow at 1pm?",
    "FREE entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121",
    "Don't forget to pick up milk on your way home. Thanks!"
];

// DOM Elements
const messageInput = document.getElementById('messageInput');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const charCount = document.getElementById('charCount');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    messageInput.addEventListener('input', updateCharCount);
    predictBtn.addEventListener('click', predictMessage);
    clearBtn.addEventListener('click', clearInput);
    
    // Enter key to predict (Ctrl+Enter)
    messageInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            predictMessage();
        }
    });
});

// Update character count
function updateCharCount() {
    const count = messageInput.value.length;
    charCount.textContent = `${count} characters`;
}

// Load sample message
function loadSample(index) {
    messageInput.value = sampleMessages[index];
    updateCharCount();
    messageInput.focus();
    
    // Smooth scroll to input
    messageInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Clear input
function clearInput() {
    messageInput.value = '';
    updateCharCount();
    results.classList.add('hidden');
    messageInput.focus();
}

// Predict message
async function predictMessage() {
    const message = messageInput.value.trim();
    
    if (!message) {
        alert('Please enter a message to analyze');
        return;
    }
    
    // Show loading, hide results
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Unknown error occurred');
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Error: ${error.message}`);
        loading.classList.add('hidden');
    } finally {
        predictBtn.disabled = false;
        clearBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    const isSpam = data.prediction === 'Spam';
    
    // Hide loading
    loading.classList.add('hidden');
    
    // Update status badge
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.className = `status-badge ${isSpam ? 'spam' : 'ham'}`;
    statusBadge.innerHTML = isSpam 
        ? '🚨 SPAM DETECTED!' 
        : '✅ LEGITIMATE MESSAGE';
    
    // Update metrics
    document.getElementById('prediction').textContent = data.prediction;
    document.getElementById('confidence').textContent = `${data.confidence}%`;
    document.getElementById('spamProb').textContent = `${data.spam_probability}%`;
    document.getElementById('hamProb').textContent = `${data.ham_probability}%`;
    
    // Update message details
    document.getElementById('msgLength').textContent = `${data.message_length} chars`;
    document.getElementById('processedLength').textContent = `${data.processed_length} chars`;
    
    // Update probability bars
    const spamBar = document.getElementById('spamBar');
    const hamBar = document.getElementById('hamBar');
    const spamPercent = document.getElementById('spamPercent');
    const hamPercent = document.getElementById('hamPercent');
    
    // Animate bars
    setTimeout(() => {
        spamBar.style.width = `${data.spam_probability}%`;
        hamBar.style.width = `${data.ham_probability}%`;
    }, 100);
    
    spamPercent.textContent = `${data.spam_probability}%`;
    hamPercent.textContent = `${data.ham_probability}%`;
    
    // Show results
    results.classList.remove('hidden');
    
    // Smooth scroll to results
    setTimeout(() => {
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

// Keyboard shortcuts info
console.log(`
📱 Spam SMS Detection - Keyboard Shortcuts
==========================================
Ctrl + Enter: Analyze message
Click sample cards to auto-fill messages

🎯 CodSoft ML Internship - Task 4
👨‍💻 Developed by Chandan Kumar
`);