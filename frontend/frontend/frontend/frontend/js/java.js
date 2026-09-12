// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Setup event listeners
    setupNavigation();
    setupQueryPage();
    setupPipeline();
    setupSidebar();
    setupTopbar();
    setupCharts();
}

// ============ NAVIGATION ============
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const pageName = this.dataset.page;
            navigateToPage(pageName);
        });
    });
}

function navigateToPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.style.display = 'none';
    });

    // Remove active class from nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });

    // Show selected page
    const selectedPage = document.getElementById(`${pageName}-page`);
    if (selectedPage) {
        selectedPage.style.display = 'block';
        selectedPage.classList.add('active');
    }

    // Activate nav link
    document.querySelector(`[data-page="${pageName}"]`).classList.add('active');

    // Update page title
    const titleMap = {
        'dashboard': 'Dashboard',
        'query': 'Live Query',
        'analytics': 'Analytics',
        'pipeline': 'Pipeline',
        'reports': 'Reports'
    };

    document.querySelector('.page-title').textContent = titleMap[pageName] || 'Dashboard';

    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        document.getElementById('sidebar').classList.remove('active');
    }
}

// ============ SIDEBAR TOGGLE ============
function setupSidebar() {
    const sidebarToggle = document.getElementById('sidebarToggle');
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.getElementById('sidebar');

    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('active');
        });
    }

    if (menuBtn) {
        menuBtn.addEventListener('click', function() {
            sidebar.classList.toggle('active');
        });
    }

    // Close sidebar when clicking outside
    document.addEventListener('click', function(e) {
        if (!sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('active');
            }
        }
    });
}

// ============ TOPBAR ============
function setupTopbar() {
    const searchInput = document.querySelector('.search-bar input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            console.log('Search:', this.value);
        });
    }
}

// ============ QUERY PAGE ============
function setupQueryPage() {
    const processBtn = document.getElementById('processBtn');
    const clearBtn = document.getElementById('clearBtn');
    const queryInput = document.getElementById('queryInput');
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');

    if (processBtn) {
        processBtn.addEventListener('click', processQuery);
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            queryInput.value = '';
            document.getElementById('resultsContainer').style.display = 'none';
            document.getElementById('historyContainer').style.display = 'none';
        });
    }

    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', function() {
            thresholdValue.textContent = this.value;
        });
    }
}

function processQuery() {
    const query = document.getElementById('queryInput').value.trim();
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);

    if (!query) {
        showToast('Please enter a query', 'warning');
        return;
    }

    showLoader(true);

    // Simulate API call
    setTimeout(() => {
        showLoader(false);

        // Generate mock result
        const result = generateMockResult(query, threshold);

        // Display results
        displayResults(result);

        // Add to history
        addToHistory(result);

        showToast('Query processed successfully', 'success');
    }, 2000);
}

function generateMockResult(query, threshold) {
    const intents = [
        "Delivery Problem",
        "Refund Request",
        "Order Delay",
        "Payment Issue",
        "Product Complaint",
        "Account Access",
        "Return Request",
        "Other"
    ];

    const intent = intents[Math.floor(Math.random() * intents.length)];
    const confidence = 0.7 + Math.random() * 0.25;

    return {
        query: query,
        intent: intent,
        confidence: confidence,
        escalation: {
            decision: confidence > threshold ? "AUTO_HANDLE" : "ESCALATE_TO_HUMAN",
            reason: confidence > threshold
                ? `Standard support inquiry within operational bounds for intent '${intent}' (confidence: ${confidence.toFixed(2)}).`
                : `Low confidence detected for intent classification (confidence: ${confidence.toFixed(2)}, threshold: ${threshold}).`
        },
        retrieved_examples: [
            {
                similarity_score: 0.86,
                intent: intent,
                customer_message: "Similar customer issue about " + intent.toLowerCase() + "...",
                agent_reply: "We apologize for the inconvenience. Let us help you resolve this issue..."
            },
            {
                similarity_score: 0.82,
                intent: intent,
                customer_message: "Another related issue...",
                agent_reply: "Thank you for reaching out. We're here to assist..."
            },
            {
                similarity_score: 0.78,
                intent: intent,
                customer_message: "Customer query about similar topic...",
                agent_reply: "We appreciate your patience. Here's what we can do..."
            }
        ],
        generated_reply: `We appreciate your message regarding "${intent.toLowerCase()}". We're committed to resolving this for you. Our team has reviewed your request and we're ready to assist. Please provide any additional details that might help us serve you better. Thank you for choosing us! ^AB`,
        timestamp: new Date().toISOString()
    };
}

function displayResults(result) {
    const resultsContainer = document.getElementById('resultsContainer');

    // Display intent and confidence
    document.getElementById('resultIntent').textContent = result.intent;
    const confidencePercent = Math.round(result.confidence * 100);
    document.getElementById('confidenceFill').style.width = confidencePercent + '%';
    document.getElementById('confidenceText').textContent = confidencePercent + '%';

    // Display escalation
    const escalationResult = document.getElementById('escalationResult');
    escalationResult.className = 'escalation-box ' + (result.escalation.decision === 'AUTO_HANDLE' ? 'escalation-auto' : 'escalation-escalate');
    escalationResult.innerHTML = `<strong>${result.escalation.decision === 'AUTO_HANDLE' ? '✅ AUTO_HANDLE' : '🚨 ESCALATE_TO_HUMAN'}</strong>`;
    document.getElementById('escalationReason').textContent = result.escalation.reason;

    // Display examples
    const examplesContainer = document.getElementById('examplesContainer');
    examplesContainer.innerHTML = '';
    result.retrieved_examples.forEach((example, index) => {
        const exampleHtml = `
            <div class="example-item">
                <div class="example-header">
                    <span>Example ${index + 1}</span>
                    <span class="example-similarity">${(example.similarity_score * 100).toFixed(0)}% Match</span>
                </div>
                <div class="example-content">
                    <div class="example-message">
                        <strong>Customer:</strong>
                        ${example.customer_message}
                    </div>
                    <div class="example-message">
                        <strong>Agent:</strong>
                        ${example.agent_reply}
                    </div>
                </div>
            </div>
        `;
        examplesContainer.innerHTML += exampleHtml;
    });

    // Display generated response
    document.getElementById('generatedResponse').textContent = result.generated_reply;

    // Show results container
    resultsContainer.style.display = 'block';
}

function addToHistory(result) {
    const historyContainer = document.getElementById('historyContainer');
    const queryHistory = document.getElementById('queryHistory');

    if (!queryHistory.children.length) {
        historyContainer.style.display = 'block';
    }

    const historyHtml = `
        <div class="history-item">
            <div class="history-item-query">${result.query.substring(0, 80)}${result.query.length > 80 ? '...' : ''}</div>
            <div class="history-item-details">
                <span><strong>Intent:</strong> ${result.intent}</span>
                <span><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</span>
                <span><strong>Escalation:</strong> ${result.escalation.decision}</span>
            </div>
        </div>
    `;

    queryHistory.insertAdjacentHTML('afterbegin', historyHtml);
}

// ============ PIPELINE ============
function setupPipeline() {
    const executeBtn = document.getElementById('executeBtn');

    if (executeBtn) {
        executeBtn.addEventListener('click', executePipeline);
    }
}

function executePipeline() {
    const mode = document.querySelector('input[name="pipeline-mode"]:checked').value;

    showLoader(true);
    const progressContainer = document.getElementById('progressContainer');
    const pipelineResults = document.getElementById('pipelineResults');

    progressContainer.style.display = 'block';
    pipelineResults.style.display = 'none';

    const steps = mode === 'full'
        ? ["Preprocessing", "Training", "Golden Dataset", "Evaluation", "Report Generation"]
        : [`${mode.charAt(0).toUpperCase() + mode.slice(1)}`];

    const progressSteps = document.getElementById('progressSteps');
    progressSteps.innerHTML = '';

    steps.forEach((step, index) => {
        const stepHtml = `
            <div class="progress-step" id="step-${index}">
                <div class="progress-step-icon">${index + 1}</div>
                <span>${step}...</span>
            </div>
        `;
        progressSteps.innerHTML += stepHtml;
    });

    let currentStep = 0;

    function executeStep() {
        if (currentStep < steps.length) {
            // Mark current step as active
            document.querySelectorAll('.progress-step').forEach((step, i) => {
                step.classList.remove('active', 'completed');
                if (i < currentStep) {
                    step.classList.add('completed');
                    step.querySelector('.progress-step-icon').innerHTML = '✓';
                } else if (i === currentStep) {
                    step.classList.add('active');
                }
            });

            // Update progress bar
            const progress = ((currentStep + 1) / steps.length) * 100;
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressPercent').textContent = Math.round(progress) + '%';

            currentStep++;
            setTimeout(executeStep, 1500);
        } else {
            // Completed
            document.querySelectorAll('.progress-step').forEach(step => {
                step.classList.remove('active');
                step.classList.add('completed');
                if (step.querySelector('.progress-step-icon').textContent !== '✓') {
                    step.querySelector('.progress-step-icon').innerHTML = '✓';
                }
            });

            document.getElementById('progressFill').style.width = '100%';
            document.getElementById('progressPercent').textContent = '100%';

            showLoader(false);

            // Show results
            setTimeout(() => {
                progressContainer.style.display = 'none';
                displayPipelineResults(mode);
                pipelineResults.style.display = 'block';
                showToast('Pipeline executed successfully!', 'success');
            }, 500);
        }
    }

    setTimeout(executeStep, 500);
}

function displayPipelineResults(mode) {
    const resultsContent = document.getElementById('resultsContent');

    const resultsMockData = {
        'full': `
            <div style="background: var(--light-bg); padding: 20px; border-radius: 8px; line-height: 1.8;">
                <p><strong>✅ Full Pipeline Completed Successfully!</strong></p>
                <p style="margin-top: 12px;">
                    <strong>Results:</strong><br>
                    • Trained Intent Model F1: 0.871<br>
                    • Golden Set Intent Accuracy: 0.878<br>
                    • Escalation Decision Accuracy: 0.92<br>
                    • Claude Judge Quality: 4.2 / 5.0<br>
                    • Failures Identified: 3
                </p>
            </div>
        `,
        'preprocess': `
            <div style="background: var(--light-bg); padding: 20px; border-radius: 8px; line-height: 1.8;">
                <p><strong>✅ Data Preprocessing Complete</strong></p>
                <p style="margin-top: 12px;">
                    • Records Processed: 1,247<br>
                    • Intent Distribution: 8 categories<br>
                    • Output: amazon_help_processed.csv
                </p>
            </div>
        `,
        'train': `
            <div style="background: var(--light-bg); padding: 20px; border-radius: 8px; line-height: 1.8;">
                <p><strong>✅ Model Training Complete</strong></p>
                <p style="margin-top: 12px;">
                    • Accuracy: 87.3%<br>
                    • Precision: 89.2%<br>
                    • Recall: 85.1%<br>
                    • F1-Score: 87.1%
                </p>
            </div>
        `,
        'golden': `
            <div style="background: var(--light-bg); padding: 20px; border-radius: 8px; line-height: 1.8;">
                <p><strong>✅ Golden Dataset Generated</strong></p>
                <p style="margin-top: 12px;">
                    • Samples Generated: 200<br>
                    • Stratified Distribution: Yes<br>
                    • Output: golden_dataset/golden_dataset.csv
                </p>
            </div>
        `,
        'evaluate': `
            <div style="background: var(--light-bg); padding: 20px; border-radius: 8px; line-height: 1.8;">
                <p><strong>✅ Evaluation Complete</strong></p>
                <p style="margin-top: 12px;">
                    • Intent Accuracy: 87.8%<br>
                    • Escalation Accuracy: 92%<br>
                    • LLM Judge Score: 4.2/5.0<br>
                    • Failure Cases: 3
                </p>
            </div>
        `
    };

    resultsContent.innerHTML = resultsMockData[mode] || '<p>Pipeline executed</p>';
}

// ============ CHARTS ============
function setupCharts() {
    createPerformanceChart();
    createIntentChart();
}

function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Majority Class', 'Keyword Rules', 'Logistic Regression'],
            datasets: [
                {
                    label: 'Accuracy',
                    data: [0.45, 0.72, 0.873],
                    backgroundColor: 'rgba(255, 153, 0, 0.8)',
                },
                {
                    label: 'Precision',
                    data: [0.45, 0.71, 0.892],
                    backgroundColor: 'rgba(20, 110, 180, 0.8)',
                },
                {
                    label: 'Recall',
                    data: [0.45, 0.70, 0.851],
                    backgroundColor: 'rgba(49, 162, 76, 0.8)',
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
}

function createIntentChart() {
    const ctx = document.getElementById('intentChart');
    if (!ctx) return;

    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [
                'Delivery Problem',
                'Refund Request',
                'Order Delay',
                'Payment Issue',
                'Return Request',
                'Account Access',
                'Product Complaint',
                'Other'
            ],
            datasets: [{
                data: [24, 18, 16, 14, 12, 10, 8, 2],
                backgroundColor: [
                    '#FF9900',
                    '#146EB4',
                    '#31A24C',
                    '#FFA500',
                    '#C7254E',
                    '#20B2AA',
                    '#FFD700',
                    '#808080'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                }
            }
        }
    });
}

// ============ UTILITIES ============
function showLoader(show) {
    const loader = document.getElementById('loader');
    if (show) {
        loader.classList.add('active');
    } else {
        loader.classList.remove('active');
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast ' + type;
    toast.style.display = 'block';

    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

// Handle window resize for responsive behavior
window.addEventListener('resize', function() {
    if (window.innerWidth > 768) {
        document.getElementById('sidebar').classList.remove('active');
    }
});