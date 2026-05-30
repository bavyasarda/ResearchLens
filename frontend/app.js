/**
 * ResearchLens - Multi-Page Dashboard Application
 * Handles navigation, API calls, and state management
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api';

// Application State
const state = {
    papers: [],
    summaries: [],
    comparison: null,
    expandedQuery: '',
    totalFetched: 0,
    history: [],
    currentPage: 'home',
    currentPaperId: null,
    papersView: 'grid',
    isSearching: false,
    hasSearched: false,
};

// DOM Elements
const elements = {};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initializeEventListeners();
    updateSliderValue();
    enableChat(false); // Start with chat disabled
});

// Initialize DOM elements
function initializeElements() {
    elements.queryInput = document.getElementById('queryInput');
    elements.numPapers = document.getElementById('numPapers');
    elements.numPapersValue = document.getElementById('numPapersValue');
    elements.yearFrom = document.getElementById('yearFrom');
    elements.yearTo = document.getElementById('yearTo');
    elements.searchBtn = document.getElementById('searchBtn');
    elements.loadingContainer = document.getElementById('loadingContainer');
    elements.loadingStatus = document.getElementById('loadingStatus');
    elements.resultsPreview = document.getElementById('resultsPreview');
    elements.papersBadge = document.getElementById('papersBadge');
    elements.papersContainer = document.getElementById('papersContainer');
    elements.papersPageDesc = document.getElementById('papersPageDesc');
    elements.summariesContainer = document.getElementById('summariesContainer');
    elements.comparisonTableContainer = document.getElementById('comparisonTableContainer');
    elements.alignmentAnalysis = document.getElementById('alignmentAnalysis');
    elements.analysisText = document.getElementById('analysisText');
    elements.chatMessages = document.getElementById('chatMessages');
    elements.chatInput = document.getElementById('chatInput');
    elements.sendBtn = document.getElementById('sendBtn');
    elements.paperHero = document.getElementById('paperHero');
    elements.aboutModal = document.getElementById('aboutModal');
    elements.paperQuestion = document.getElementById('paperQuestion');
}

// Initialize event listeners
function initializeEventListeners() {
    // Slider
    elements.numPapers.addEventListener('input', updateSliderValue);

    // Search
    elements.searchBtn.addEventListener('click', performSearch);
    elements.queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            performSearch();
        }
    });

    // Navigation
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            if (page === 'chat' && !state.hasSearched) return;
            navigateTo(page);
        });
    });

    // About modal
    document.getElementById('aboutBtn').addEventListener('click', showModal);
    document.querySelector('.close-btn').addEventListener('click', closeModal);
    document.getElementById('aboutModal').addEventListener('click', (e) => {
        if (e.target.id === 'aboutModal') closeModal();
    });

    // Chat
    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    elements.sendBtn.addEventListener('click', sendChatMessage);

    // Paper question
    elements.paperQuestion.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askAboutPaper();
        }
    });
}

// Update slider display
function updateSliderValue() {
    elements.numPapersValue.textContent = elements.numPapers.value;
}

// Navigate to page
function navigateTo(page) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.style.display = 'none');

    // Update nav items
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });

    // Show target page
    const targetPage = document.getElementById(`${page}Page`);
    if (targetPage) {
        targetPage.style.display = 'block';
        targetPage.style.animation = 'fadeIn 0.3s ease';
    }

    state.currentPage = page;

    // Scroll to top
    window.scrollTo(0, 0);
}

// Enable/disable chat
function enableChat(enabled) {
    elements.chatInput.disabled = !enabled;
    elements.sendBtn.disabled = !enabled;
}

// Perform search
async function performSearch() {
    const query = elements.queryInput.value.trim();
    if (!query) {
        showToast('Please enter a search query.', 'error');
        return;
    }

    if (state.isSearching) return;

    const request = {
        query: query,
        num_papers: parseInt(elements.numPapers.value),
        preference: document.querySelector('input[name="preference"]:checked').value,
        year_from: elements.yearFrom.value ? parseInt(elements.yearFrom.value) : null,
        year_to: elements.yearTo.value ? parseInt(elements.yearTo.value) : null,
    };

    // Show loading
    setLoadingState(true);
    resetProgressSteps();
    updateProgressStep(1, 'active');
    elements.loadingStatus.textContent = 'Expanding query...';
    elements.loadingContainer.style.display = 'block';
    elements.resultsPreview.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Update state
        state.papers = data.papers || [];
        state.summaries = data.summaries || [];
        state.comparison = data.comparison || null;
        state.expandedQuery = data.expanded_query || query;
        state.totalFetched = data.total_fetched || 0;
        state.history = [];
        state.hasSearched = true;

        // Mark all steps complete
        for (let i = 1; i <= 5; i++) updateProgressStep(i, 'completed');
        elements.loadingStatus.textContent = 'Complete!';

        // Update badge
        elements.papersBadge.textContent = state.papers.length;

        // Show results preview
        setTimeout(() => {
            elements.loadingContainer.style.display = 'none';
            showResultsPreview();
            // Enable chat
            enableChat(true);
            showToast(`Found ${state.papers.length} papers!`, 'success');
        }, 800);

    } catch (error) {
        console.error('Search failed:', error);
        showToast('Search failed: ' + error.message, 'error');
        elements.loadingContainer.style.display = 'none';
    } finally {
        setLoadingState(false);
    }
}

// Show results preview
function showResultsPreview() {
    document.getElementById('statPapers').textContent = state.papers.length;
    document.getElementById('statSummaries').textContent = state.summaries.length;

    const avgRelevance = state.summaries.length > 0
        ? Math.round(state.summaries.reduce((acc, s) => acc + s.relevance_score, 0) / state.summaries.length * 100)
        : 0;
    document.getElementById('statAvgRelevance').textContent = avgRelevance + '%';

    elements.resultsPreview.style.display = 'block';
    elements.resultsPreview.style.animation = 'slideUp 0.4s ease';

    // Show expanded query if available
    if (state.expandedQuery && state.expandedQuery !== elements.queryInput.value) {
        showToast(`Query expanded: "${truncateText(state.expandedQuery, 60)}..."`, 'info');
    }
}

// Reset progress steps
function resetProgressSteps() {
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        if (step) {
            step.className = 'progress-step';
            step.querySelector('.step-indicator').textContent = i;
        }
    }
}

// Update progress step
function updateProgressStep(stepNum, status) {
    const step = document.getElementById(`step${stepNum}`);
    if (step) {
        step.className = `progress-step ${status}`;
        const indicator = step.querySelector('.step-indicator');
        if (status === 'completed') {
            indicator.innerHTML = '✓';
        } else {
            indicator.textContent = stepNum;
        }
    }
}

// Set loading state
function setLoadingState(isLoading) {
    state.isSearching = isLoading;
    elements.searchBtn.disabled = isLoading;
}

// Set papers view
function setPapersView(view) {
    state.papersView = view;
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });
    elements.papersContainer.classList.toggle('list-view', view === 'list');
}

// Render papers
function renderPapers() {
    elements.papersPageDesc.textContent = `Found ${state.papers.length} papers matching your query`;

    if (state.papers.length === 0) {
        elements.papersContainer.innerHTML = renderEmptyState('No papers found');
        return;
    }

    elements.papersContainer.innerHTML = state.papers.map((paper, index) => {
        const summary = state.summaries.find(s => s.paper_id === paper.paper_id);
        const relevance = summary ? summary.relevance_score : 0;
        const relevanceClass = relevance >= 0.7 ? 'high' : relevance >= 0.4 ? 'medium' : 'low';
        const authors = paper.authors.length > 0
            ? paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' et al.' : '')
            : 'Unknown';

        // Display venue properly - don't show "arXiv" as venue
        const displayVenue = paper.venue && paper.venue !== 'arXiv' ? paper.venue : (paper.venue === 'arXiv' ? 'Preprint' : '');

        return `
            <div class="paper-card" onclick="openPaperDetail('${paper.paper_id}')" style="animation-delay: ${index * 50}ms">
                <div class="paper-header">
                    <h3 class="paper-title">${escapeHtml(paper.title)}</h3>
                    <span class="source-badge ${paper.source}">${paper.source === 'arxiv' ? 'arXiv' : paper.source}</span>
                </div>
                <div class="paper-meta">
                    <span>👤 ${escapeHtml(authors)}</span>
                    <span>📅 ${paper.year}</span>
                    ${displayVenue ? `<span>📖 ${escapeHtml(displayVenue)}</span>` : ''}
                    ${paper.citation_count > 0 ? `<span>📊 ${paper.citation_count.toLocaleString()}</span>` : ''}
                </div>
                <p class="paper-abstract">${escapeHtml(truncateText(paper.abstract || 'No abstract available.', 200))}</p>
                <div class="relevance-bar">
                    <span class="relevance-label">Relevance</span>
                    <div class="relevance-track">
                        <div class="relevance-fill ${relevanceClass}" style="width: ${relevance * 100}%"></div>
                    </div>
                    <span class="relevance-score">${Math.round(relevance * 100)}%</span>
                </div>
                <div class="paper-actions" onclick="event.stopPropagation()">
                    <button class="paper-btn" onclick="openPaperDetail('${paper.paper_id}')">View Details</button>
                    ${paper.url ? `<a class="paper-btn" href="${paper.url}" target="_blank">View Paper</a>` : ''}
                    ${paper.pdf_url ? `<a class="paper-btn" href="${paper.pdf_url}" target="_blank">PDF</a>` : ''}
                </div>
            </div>
        `;
    }).join('');
}

// Open paper detail
function openPaperDetail(paperId) {
    state.currentPaperId = paperId;
    const paper = state.papers.find(p => p.paper_id === paperId);
    const summary = state.summaries.find(s => s.paper_id === paperId);

    if (!paper) {
        showToast('Paper not found', 'error');
        return;
    }

    // Update header
    elements.paperHero.innerHTML = `
        <h1>${escapeHtml(paper.title)}</h1>
    `;

    // Update sections with better formatting
    document.getElementById('paperAbstract').textContent = paper.abstract || 'No abstract available.';

    // Summary with fallback
    const summaryText = summary?.summary || 'Summary not available yet.';
    document.getElementById('paperSummary').textContent = summaryText;

    // Methodology with fallback
    const methodologyText = (summary?.key_methodology && !summary.key_methodology.includes('Methodology described'))
        ? summary.key_methodology : 'Methodology details extracted from paper content.';
    document.getElementById('paperMethodology').textContent = methodologyText;

    // Findings with fallback
    const findingsText = (summary?.key_findings && !summary.key_findings.includes('See paper'))
        ? summary.key_findings : 'Key findings extracted from paper content.';
    document.getElementById('paperFindings').textContent = findingsText;

    // Update sidebar
    const authors = paper.authors.length > 0 ? paper.authors.join(', ') : 'Unknown';
    document.getElementById('detailAuthors').textContent = authors;
    document.getElementById('detailYear').textContent = paper.year;

    // Venue display
    const displayVenue = paper.venue && paper.venue !== 'arXiv' ? paper.venue : (paper.venue === 'arXiv' ? 'Preprint / arXiv' : 'Unknown');
    document.getElementById('detailVenue').textContent = displayVenue;

    document.getElementById('detailCitations').textContent = paper.citation_count > 0 ? paper.citation_count.toLocaleString() : 'N/A';
    document.getElementById('detailSource').textContent = paper.source === 'arxiv' ? 'arXiv' : paper.source;

    // Update relevance
    const relDisplay = document.getElementById('detailRelevance');
    const relScore = summary?.relevance_score || 0;
    const relClass = relScore >= 0.7 ? 'high' : relScore >= 0.4 ? 'medium' : 'low';
    relDisplay.innerHTML = `
        <div class="relevance-score ${relClass}">${Math.round(relScore * 100)}%</div>
        <div class="relevance-bar">
            <div class="relevance-fill ${relClass}" style="width: ${relScore * 100}%"></div>
        </div>
        <p class="relevance-reason">${escapeHtml(summary?.relevance_reason || 'Relevance to your query')}</p>
    `;

    // Update links
    document.getElementById('paperUrl').href = paper.url || '#';
    document.getElementById('paperPdfUrl').href = paper.pdf_url || '#';

    // Clear paper question
    elements.paperQuestion.value = '';

    navigateTo('paperDetail');
}

// Ask about current paper
async function askAboutPaper() {
    const question = elements.paperQuestion.value.trim();
    if (!question) {
        showToast('Please enter a question', 'error');
        return;
    }

    if (state.currentPage !== 'paperDetail') {
        showToast('Please view a paper first', 'error');
        return;
    }

    const paper = state.papers.find(p => p.paper_id === state.currentPaperId);
    if (!paper) {
        showToast('Paper not found', 'error');
        return;
    }

    // Create focused question for this paper
    const fullQuestion = `[About paper: "${truncateText(paper.title, 50)}..."]\n\n${question}`;

    // Navigate to chat and send message
    navigateTo('chat');
    elements.chatInput.value = fullQuestion;
    await sendChatMessage();
    elements.paperQuestion.value = '';
}

// Render summaries
function renderSummaries() {
    if (state.summaries.length === 0) {
        elements.summariesContainer.innerHTML = renderEmptyState('No summaries available');
        return;
    }

    const paperMap = {};
    state.papers.forEach(p => paperMap[p.paper_id] = p);

    elements.summariesContainer.innerHTML = state.summaries.map((summary, index) => {
        const paper = paperMap[summary.paper_id];
        const relClass = summary.relevance_score >= 0.7 ? 'high' : summary.relevance_score >= 0.4 ? 'medium' : 'low';
        const title = paper ? truncateText(paper.title, 80) : `Paper ${index + 1}`;
        const author = paper?.authors[0] || 'Unknown';

        return `
            <div class="summary-card" onclick="toggleSummary(${index})" style="animation-delay: ${index * 100}ms">
                <div class="summary-header">
                    <div class="summary-header-left">
                        <h3 class="summary-title">${escapeHtml(title)}</h3>
                        <div class="summary-meta">
                            <span>${paper?.year || '-'} • ${escapeHtml(author)} • ${Math.round(summary.relevance_score * 100)}% match</span>
                        </div>
                    </div>
                    <span class="summary-toggle">▼</span>
                </div>
                <div class="summary-content">
                    <div class="summary-section">
                        <div class="summary-label">📝 AI Summary</div>
                        <p class="summary-text">${escapeHtml(summary.summary || 'Summary not available')}</p>
                    </div>
                    <div class="summary-section">
                        <div class="summary-label">🔬 Key Methodology</div>
                        <p class="summary-text">${escapeHtml(summary.key_methodology || 'Methodology not extracted')}</p>
                    </div>
                    <div class="summary-section">
                        <div class="summary-label">📊 Key Findings</div>
                        <p class="summary-text">${escapeHtml(summary.key_findings || 'Findings not extracted')}</p>
                    </div>
                    <div class="summary-section">
                        <div class="summary-label">🎯 Relevance Assessment</div>
                        <div class="relevance-bar" style="margin-bottom: 12px;">
                            <div class="relevance-track" style="flex: 1;">
                                <div class="relevance-fill ${relClass}" style="width: ${summary.relevance_score * 100}%"></div>
                            </div>
                        </div>
                        <p class="relevance-reason">${escapeHtml(summary.relevance_reason || '')}</p>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Toggle summary
window.toggleSummary = function(index) {
    const card = document.querySelectorAll('.summary-card')[index];
    if (card) card.classList.toggle('open');
};

// Render comparison
function renderComparison() {
    if (!state.comparison || !state.comparison.rows || state.comparison.rows.length === 0) {
        elements.comparisonTableContainer.innerHTML = renderEmptyState('No comparison available');
        return;
    }

    elements.comparisonTableContainer.innerHTML = `
        <table class="comparison-table">
            <thead>
                <tr>
                    ${state.comparison.headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
                ${state.comparison.rows.map(row => `
                    <tr>
                        ${state.comparison.headers.map(h => {
                            let value = row[h] || '—';
                            if (h === 'Relevance' || h === 'Alignment') {
                                const level = value.split('—')[0].trim().toLowerCase();
                                const cls = level.includes('high') ? 'high' : level.includes('medium') ? 'medium' : 'low';
                                return `<td><span class="alignment-cell ${cls}">${escapeHtml(value)}</span></td>`;
                            }
                            return `<td>${escapeHtml(String(value))}</td>`;
                        }).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    document.getElementById('analysisText').textContent = state.comparison.alignment_analysis || 'Analysis not available.';
}

// Send chat message
async function sendChatMessage() {
    const message = elements.chatInput.value.trim();
    if (!message || elements.chatInput.disabled) return;
    await sendChatMessageWithText(message);
}

// Send chat with specific text
async function sendChatMessageWithText(message) {
    // Add user message
    addChatMessage('user', message);
    elements.chatInput.value = '';
    state.history.push({ role: 'user', content: message });

    // Add loading indicator
    const loadingEl = addChatMessage('assistant', 'Thinking...');

    try {
        const request = {
            message: message,
            context_papers: state.papers,
            context_summaries: state.summaries,
            history: state.history.slice(-20), // Limit history to last 20 messages
        };

        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });

        console.log('Response status:', response.status);
        const rawText = await response.text();
        console.log('Raw response:', rawText);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${rawText}`);
        }

        let data;
        try {
            data = JSON.parse(rawText);
        } catch (e) {
            console.error('JSON parse error:', e);
            data = { response: rawText };
        }

        console.log('Parsed data:', data);

        // Check for the actual response field
        let content = '';
        if (typeof data === 'string') {
            content = data;
        } else if (data.response) {
            content = data.response;
        } else if (data.content) {
            content = data.content;
        } else if (data.message) {
            content = data.message;
        } else {
            content = JSON.stringify(data);
        }

        if (!content || content === '{}') {
            content = 'I apologize, but I encountered an error.';
        }

        console.log('Final content:', content.substring(0, 100));

        // Update loading message with response
        const contentEl = loadingEl.querySelector('.message-content');
        if (contentEl) {
            contentEl.innerHTML = formatMarkdown(content);
        }

        state.history.push({ role: 'assistant', content: content });

    } catch (error) {
        console.error('Chat failed:', error);
        const contentEl = loadingEl.querySelector('.message-content');
        contentEl.textContent = 'Sorry, I encountered an error. Please try again.';
        showToast('Chat error: ' + error.message, 'error');
    }
}

// Add chat message
function addChatMessage(role, content) {
    // Remove empty state if present
    const empty = elements.chatMessages.querySelector('.chat-empty');
    if (empty) empty.remove();

    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${role}`;

    const avatar = role === 'user' ? 'U' : 'AI';

    messageEl.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;

    elements.chatMessages.appendChild(messageEl);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    return messageEl;
}

// Format markdown in chat
function formatMarkdown(text) {
    if (!text) return '';
    let formatted = escapeHtml(text);

    // Bold
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Headers
    formatted = formatted.replace(/^### (.*$)/gm, '<h4>$1</h4>');
    formatted = formatted.replace(/^## (.*$)/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^# (.*$)/gm, '<h2>$1</h2>');
    // Lists
    formatted = formatted.replace(/^- (.*$)/gm, '<li>$1</li>');
    formatted = formatted.replace(/^\d+\. (.*$)/gm, '<li>$1</li>');
    formatted = formatted.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    // Line breaks
    formatted = formatted.replace(/\n\n/g, '</p><p>');
    formatted = formatted.replace(/\n/g, '<br>');

    return `<p>${formatted}</p>`;
}

// Ask suggestion
window.askSuggestion = async function(text) {
    elements.chatInput.value = text;
    await sendChatMessage();
};

// Clear chat
function clearChat() {
    state.history = [];
    elements.chatMessages.innerHTML = `
        <div class="chat-empty">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <p>Ask me anything about the papers!</p>
            <div class="suggestions">
                <button onclick="askSuggestion('Which paper is most relevant to my query?')">Most relevant?</button>
                <button onclick="askSuggestion('Compare the methodologies of all papers')">Compare methods</button>
                <button onclick="askSuggestion('What are the main limitations?')">Limitations?</button>
            </div>
        </div>
    `;
}

// Show toast notification
function showToast(message, type = 'info') {
    // Remove existing toast
    const existingToast = document.querySelector('.toast');
    if (existingToast) existingToast.remove();

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span>${escapeHtml(message)}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;

    // Add styles if not exist
    if (!document.querySelector('#toast-styles')) {
        const style = document.createElement('style');
        style.id = 'toast-styles';
        style.textContent = `
            .toast {
                position: fixed;
                bottom: 24px;
                right: 24px;
                padding: 16px 20px;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-md);
                display: flex;
                align-items: center;
                gap: 12px;
                z-index: 1001;
                animation: slideIn 0.3s ease;
                box-shadow: var(--shadow-lg);
            }
            .toast-success { border-left: 4px solid var(--accent-green); }
            .toast-error { border-left: 4px solid var(--accent-red); }
            .toast-info { border-left: 4px solid var(--accent-cyan); }
            .toast button {
                background: none;
                border: none;
                color: var(--text-muted);
                cursor: pointer;
                font-size: 1.25rem;
            }
            @keyframes slideIn {
                from { opacity: 0; transform: translateX(100%); }
                to { opacity: 1; transform: translateX(0); }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(toast);

    // Auto remove after 4 seconds
    setTimeout(() => {
        if (toast.parentElement) toast.remove();
    }, 4000);
}

// Show modal
function showModal() {
    elements.aboutModal.classList.add('show');
}

// Close modal
function closeModal() {
    elements.aboutModal.classList.remove('show');
}

// Render empty state
function renderEmptyState(message) {
    return `
        <div class="empty-state" style="text-align: center; padding: 60px 20px;">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="opacity: 0.3; margin-bottom: 16px;">
                <circle cx="11" cy="11" r="8"/>
                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <p style="color: var(--text-muted); font-size: 1rem;">${message}</p>
        </div>
    `;
}

// Utility functions
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
}

// Page-specific render functions
const pageRenderers = {
    papers: renderPapers,
    summaries: renderSummaries,
    compare: renderComparison,
};

// Override navigateTo to render page content
const originalNavigateTo = navigateTo;
navigateTo = function(page) {
    originalNavigateTo(page);

    // Render page content if we have data
    if (state.hasSearched && pageRenderers[page]) {
        pageRenderers[page]();
    }

    // Handle chat page
    if (page === 'chat' && !state.hasSearched) {
        navigateTo('home');
        showToast('Please search for papers first', 'info');
    }
};

// Make functions global
window.navigateTo = navigateTo;
window.setPapersView = setPapersView;
window.openPaperDetail = openPaperDetail;
window.askAboutPaper = askAboutPaper;
window.askSuggestion = window.askSuggestion;
window.clearChat = clearChat;
window.showModal = showModal;
window.closeModal = closeModal;
window.toggleSummary = toggleSummary;