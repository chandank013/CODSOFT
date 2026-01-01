// DOM Elements
const seedText = document.getElementById('seedText');
const modelType = document.getElementById('modelType');
const genLength = document.getElementById('genLength');
const temperature = document.getElementById('temperature');
const lengthValue = document.getElementById('lengthValue');
const tempValue = document.getElementById('tempValue');
const generateBtn = document.getElementById('generateBtn');
const copyBtn = document.getElementById('copyBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Update range displays
    genLength.addEventListener('input', updateLengthDisplay);
    temperature.addEventListener('input', updateTempDisplay);
    
    // Generate button
    generateBtn.addEventListener('click', generateText);
    
    // Copy button
    copyBtn.addEventListener('click', copyToClipboard);
    
    // Initialize displays
    updateLengthDisplay();
    updateTempDisplay();
    
    console.log('✍️ Handwritten Text Generation - Ready!');
});

// Update length display
function updateLengthDisplay() {
    const value = genLength.value;
    lengthValue.textContent = `${value} characters`;
}

// Update temperature display
function updateTempDisplay() {
    const value = parseFloat(temperature.value);
    let desc = 'Balanced';
    
    if (value < 0.5) desc = 'Conservative';
    else if (value > 1.0) desc = 'Creative';
    
    tempValue.textContent = `${value} (${desc})`;
}

// Load sample seed
function loadSeed(seed) {
    seedText.value = seed;
    seedText.focus();
    seedText.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Generate text
async function generateText() {
    const seed = seedText.value.trim();
    
    if (!seed) {
        alert('Please enter seed text');
        return;
    }
    
    // Show loading
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    generateBtn.disabled = true;
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                seed_text: seed,
                model: modelType.value,
                length: parseInt(genLength.value),
                temperature: parseFloat(temperature.value)
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Generation failed');
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Unknown error occurred');
        }
        
    } catch (error) {
        console.error('Generation error:', error);
        alert(`Error: ${error.message}`);
        loading.classList.add('hidden');
    } finally {
        generateBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    // Hide loading
    loading.classList.add('hidden');
    
    // Update metrics
    const modelNames = {
        'rnn': 'Simple RNN',
        'lstm': 'LSTM',
        'gru': 'GRU'
    };
    
    document.getElementById('usedModel').textContent = modelNames[data.model] || data.model.toUpperCase();
    document.getElementById('generatedLength').textContent = `${data.length} chars`;
    document.getElementById('usedTemp').textContent = data.temperature.toFixed(1);
    document.getElementById('totalLength').textContent = `${data.total_length} chars`;
    
    // Display seed and generated text
    document.getElementById('displaySeed').textContent = data.seed_text;
    
    // Extract only the generated portion (remove seed)
    const generatedOnly = data.generated_text.substring(data.seed_text.length);
    document.getElementById('generatedText').textContent = generatedOnly;
    
    // Show results
    results.classList.remove('hidden');
    
    // Scroll to results
    setTimeout(() => {
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

// Copy to clipboard
function copyToClipboard() {
    const seed = document.getElementById('displaySeed').textContent;
    const generated = document.getElementById('generatedText').textContent;
    const fullText = seed + generated;
    
    navigator.clipboard.writeText(fullText).then(() => {
        // Change button text temporarily
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '✅ Copied!';
        copyBtn.style.background = '#10b981';
        
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
            copyBtn.style.background = '';
        }, 2000);
    }).catch(err => {
        console.error('Copy failed:', err);
        alert('Failed to copy text');
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        generateText();
    }
});

// Console info
console.log(`
✍️ Handwritten Text Generation - Keyboard Shortcuts
====================================================
Ctrl/Cmd + Enter: Generate text
Click sample cards to load seeds

🤖 Models Available:
• Simple RNN - Basic recurrent network
• LSTM - Long Short-Term Memory (recommended)
• GRU - Gated Recurrent Unit (fast)

🌡️ Temperature Guide:
• 0.2-0.5: Conservative and predictable
• 0.6-0.9: Balanced (recommended)
• 1.0-1.5: Creative and random

🎯 CodSoft ML Internship - Task 5
👨‍💻 Developed by Chandan Kumar
====================================================
`);