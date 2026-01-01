
const examples = [
    "A team of astronauts embarks on a dangerous mission to explore a distant planet and encounter alien life forms threatening their survival.",
    "A young woman falls in love with a stranger she meets in a coffee shop in Paris, leading to a beautiful romance that changes both their lives.",
    "A detective investigates a series of murders in a small town, uncovering dark secrets and shocking truths about the community."
];

function fillExample(index) {
    document.getElementById("movieDescription").value = examples[index];
    document.getElementById("movieDescription").focus();
}

async function predictGenre() {
    const description = document.getElementById("movieDescription").value.trim();

    if (!description) {
        alert("Please enter a movie description first!");
        return;
    }

    // UI state
    document.getElementById("loading").style.display = "block";
    document.getElementById("result").style.display = "none";
    document.getElementById("predictBtn").disabled = true;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ description })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || "Prediction failed");
        }

        document.getElementById("genreValue").textContent = data.genre;
        document.getElementById("confidenceValue").textContent =
            data.confidence ? (data.confidence * 100).toFixed(2) + "%" : "N/A";

        document.getElementById("result").style.display = "block";
    } catch (error) {
        alert("❌ Error: " + error.message);
    } finally {
        document.getElementById("loading").style.display = "none";
        document.getElementById("predictBtn").disabled = false;
    }
}

// Ctrl + Enter support
document.getElementById("movieDescription").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && e.ctrlKey) {
        predictGenre();
    }
});
