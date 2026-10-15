/**
 * GrievanceGPT Client-Side Dashboard Controller.
 * Handles interactive chat, real-time ML prediction display, completeness meter,
 * grievance editing & confirmation, RAG search, registry table, and Chart.js analytics.
 */

let currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
let currentExtractedData = {};
let currentClassification = {};
let currentDraftToken = null;
let deptChart = null;
let statusChart = null;
let catChart = null;
let urgChart = null;
let trendChart = null;
let mlBenchmarkChart = null;
let adminSessionPin = null;          // set after successful PIN verification
let currentAdminGrievanceId = null;  // ID of the complaint open in the admin drawer

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.nav-tab').forEach(el => {
        el.classList.remove('nav-tab-active');
        el.classList.add('text-slate-500', 'hover:bg-slate-50', 'hover:text-slate-900');
    });

    const targetTab = document.getElementById('tab-' + tabId);
    const targetNav = document.getElementById('nav-' + tabId);

    if (targetTab) targetTab.classList.remove('hidden');
    if (targetNav) {
        targetNav.classList.remove('text-slate-500', 'hover:bg-slate-50', 'hover:text-slate-900');
        targetNav.classList.add('nav-tab-active');
    }

    if (tabId === 'admin' && adminSessionPin) {
        loadAdminGrievances();
        loadAnalytics();
    }
    if (window.lucide) lucide.createIcons();
}

// Check System Status on Load
async function checkSystemHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const badge = document.getElementById('ollama-status-text');
        if (data.ollama && data.ollama.connected) {
            badge.textContent = `Ollama (${data.ollama.active_model}): Active`;
        } else {
            badge.textContent = `Ollama: Offline (Adaptive Fallback Active)`;
        }
    } catch (e) {
        console.warn("Health check error:", e);
    }
}

// Fill Quick Prompt
function fillPrompt(text) {
    const input = document.getElementById('chat-input');
    input.value = text;
    document.getElementById('chat-form').dispatchEvent(new Event('submit'));
}

// Reset Conversation
function resetConversation() {
    currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
    currentExtractedData = {};
    currentClassification = {};
    currentDraftToken = null;

    const container = document.getElementById('chat-messages');
    if (container) {
        container.innerHTML = `
            <div class="flex items-start space-x-3">
                <div class="w-9 h-9 rounded-2xl bg-slate-900 flex items-center justify-center flex-shrink-0 text-white shadow-sm">
                    <i data-lucide="bot" class="w-4 h-4"></i>
                </div>
                <div class="chat-bubble-ai p-4 max-w-[85%] text-sm leading-relaxed" style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; background-color: #f8fafc !important; border: 1.5px solid #94a3b8 !important;">
                    <p class="font-extrabold text-black mb-1" style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-weight: 800 !important;">Vanakkam! Welcome to GrievanceGPT.</p>
                    <p class="text-black font-bold" style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-weight: 700 !important;">
                        Please describe your public grievance in English, தமிழ், or Tanglish. I will help verify the required details and prepare your formal complaint.
                    </p>
                </div>
            </div>
        `;
    }

    if (document.getElementById('completeness-bar')) {
        updateCompletenessUI(0, []);
    }
    const elIntent = document.getElementById('res-intent');
    if (elIntent) elIntent.textContent = '-';
    const elCat = document.getElementById('res-category');
    if (elCat) elCat.textContent = '-';
    const elDept = document.getElementById('res-department');
    if (elDept) elDept.textContent = '-';
    const elUrg = document.getElementById('res-urgency');
    if (elUrg) elUrg.textContent = '-';

    const guideCard = document.getElementById('draft-info-guide');
    if (guideCard) guideCard.classList.remove('hidden');

    const draftCard = document.getElementById('draft-grievance-card');
    if (draftCard) draftCard.classList.add('hidden');

    const successBox = document.getElementById('draft-success-box');
    if (successBox) successBox.classList.add('hidden');

    const confirmBtn = document.getElementById('confirm-btn');
    if (confirmBtn) {
        confirmBtn.disabled = false;
        confirmBtn.className = 'flex-1 py-3 rounded-2xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold flex items-center justify-center space-x-1.5 shadow-sm transition';
        confirmBtn.innerHTML = `<i data-lucide="check" class="w-4 h-4 text-emerald-400"></i><span>Confirm & Submit Complaint</span>`;
    }

    if (window.lucide) lucide.createIcons();
}

// Handle Chat Message
async function handleChatSubmit(event) {
    event.preventDefault();
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    const submitBtn = document.getElementById('chat-submit-btn');
    submitBtn.disabled = true;

    // Append User Message to UI
    appendChatMessage('user', message);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId,
                accumulated_data: currentExtractedData
            })
        });

        const data = await response.json();

        // Update stored states
        currentExtractedData = data.extracted_entities || {};
        currentClassification = data.classification || {};
        currentDraftToken = data.reference_id;

        // Append Assistant Reply
        appendChatMessage('assistant', data.assistant_reply);

        // Update Classification Panel if present
        if (data.classification) {
            const elIntent = document.getElementById('res-intent');
            if (elIntent) elIntent.textContent = data.classification.intent || '-';
            const elCat = document.getElementById('res-category');
            if (elCat) elCat.textContent = (data.classification.category || '-').replace('_', ' ');
            const elDept = document.getElementById('res-department');
            if (elDept) elDept.textContent = data.classification.department || '-';
            const elUrg = document.getElementById('res-urgency');
            if (elUrg) elUrg.textContent = (data.classification.urgency || '-').toUpperCase();
        }

        // Update Completeness if element exists
        if (data.completeness && document.getElementById('completeness-bar')) {
            updateCompletenessUI(data.completeness.score, data.completeness.missing_fields);
        }

        // Handle Generated Grievance Card
        if (data.structured_grievance) {
            populateDraftCard(data.structured_grievance);
        }

    } catch (err) {
        console.error(err);
        appendChatMessage('assistant', "Sorry, an error occurred while processing your message. Please try again.");
    } finally {
        submitBtn.disabled = false;
        input.focus();
    }
}

function appendChatMessage(role, text) {
    const container = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = "flex items-start space-x-3 " + (role === 'user' ? 'justify-end' : '');

    if (role === 'user') {
        msgDiv.innerHTML = `
            <div class="chat-bubble-user p-3.5 max-w-[80%] text-sm leading-relaxed shadow-sm">
                <p class="whitespace-pre-line font-medium">${escapeHtml(text)}</p>
            </div>
            <div class="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center flex-shrink-0 text-white text-xs font-semibold">
                You
            </div>
        `;
    } else {
        msgDiv.innerHTML = `
            <div class="w-9 h-9 rounded-2xl bg-slate-900 flex items-center justify-center flex-shrink-0 text-white shadow-sm">
                <i data-lucide="bot" class="w-4 h-4"></i>
            </div>
            <div class="chat-bubble-ai p-4 max-w-[85%] text-sm leading-relaxed shadow-xs" style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; background-color: #f8fafc !important; border: 1.5px solid #94a3b8 !important;">
                <p class="whitespace-pre-line text-black font-bold" style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; opacity: 1 !important; font-weight: 700 !important;">${formatAssistantText(text)}</p>
            </div>
        `;
    }

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    if (window.lucide) lucide.createIcons();
}

function formatAssistantText(text) {
    if (!text) return '';
    return text.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-weight: 800 !important;">$1</strong>');
}

function escapeHtml(string) {
    return String(string).replace(/[&<>"']/g, function(s) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[s];
    });
}

function updateCompletenessUI(score, missingFields) {
    const bar = document.getElementById('completeness-bar');
    const badge = document.getElementById('completeness-score-badge');
    const missingBox = document.getElementById('missing-info-box');
    const missingText = document.getElementById('missing-items-text');

    if (!bar || !badge) return;

    bar.style.width = score + '%';
    badge.textContent = score + '%';

    if (score < 40) {
        bar.className = 'h-2.5 rounded-full progress-fill bg-rose-500';
    } else if (score < 70) {
        bar.className = 'h-2.5 rounded-full progress-fill bg-amber-500';
    } else {
        bar.className = 'h-2.5 rounded-full progress-fill bg-emerald-500';
    }

    if (missingBox && missingText) {
        if (missingFields && missingFields.length > 0) {
            missingBox.classList.remove('hidden');
            missingText.textContent = missingFields.join(', ');
        } else {
            missingBox.classList.add('hidden');
        }
    }
}

function populateDraftCard(draft) {
    const guideCard = document.getElementById('draft-info-guide');
    if (guideCard) guideCard.classList.add('hidden');
    const oldPlaceholder = document.getElementById('draft-placeholder-card');
    if (oldPlaceholder) oldPlaceholder.classList.add('hidden');

    const card = document.getElementById('draft-grievance-card');
    if (card) card.classList.remove('hidden');

    const tokenEl = document.getElementById('draft-token');
    if (tokenEl) tokenEl.textContent = draft.grievance_id || currentDraftToken || 'DRAFT';

    const deptBadge = document.getElementById('draft-dept-badge');
    if (deptBadge) deptBadge.textContent = draft.department || currentClassification.department || 'Municipal Administration';

    const catBadge = document.getElementById('draft-cat-badge');
    if (catBadge) catBadge.textContent = (draft.category || currentClassification.category || 'civic_issue').replace(/_/g, ' ');

    const urgBadge = document.getElementById('draft-urgency-badge');
    if (urgBadge) {
        const u = (draft.urgency || currentClassification.urgency || 'medium').toUpperCase();
        urgBadge.textContent = `${u} PRIORITY`;
        if (u === 'HIGH' || u === 'CRITICAL') {
            urgBadge.className = 'px-2.5 py-1 rounded-full bg-rose-50 text-rose-900 border border-rose-200 text-[10px] font-extrabold uppercase';
        } else if (u === 'LOW') {
            urgBadge.className = 'px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-900 border border-emerald-200 text-[10px] font-extrabold uppercase';
        } else {
            urgBadge.className = 'px-2.5 py-1 rounded-full bg-amber-50 text-amber-900 border border-amber-200 text-[10px] font-extrabold uppercase';
        }
    }

    const subInput = document.getElementById('draft-subject');
    if (subInput) subInput.value = draft.subject || '';

    const descInput = document.getElementById('draft-description');
    if (descInput) descInput.value = draft.description || '';

    const locInput = document.getElementById('draft-location');
    if (locInput) locInput.value = draft.location || '';

    const durInput = document.getElementById('draft-duration');
    if (durInput) durInput.value = draft.duration || '';

    currentDraftToken = draft.grievance_id;
    if (window.lucide) lucide.createIcons();
}

// Regenerate Grievance
async function regenerateGrievance() {
    try {
        const response = await fetch('/api/grievance/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                extracted_data: currentExtractedData,
                classification: currentClassification,
                grievance_id: currentDraftToken
            })
        });
        const data = await response.json();
        populateDraftCard(data);
    } catch (e) {
        console.error("Regeneration error:", e);
    }
}

// Confirm Grievance
async function confirmGrievance() {
    const confirmBtn = document.getElementById('confirm-btn');
    if (confirmBtn) {
        confirmBtn.disabled = true;
        confirmBtn.innerHTML = `<span>Saving...</span>`;
    }

    const payload = {
        grievance_id: currentDraftToken,
        subject: document.getElementById('draft-subject')?.value || '',
        description: document.getElementById('draft-description')?.value || '',
        location: document.getElementById('draft-location')?.value || '',
        district: currentExtractedData.district || '',
        duration: document.getElementById('draft-duration')?.value || '',
        category: currentClassification.category || currentExtractedData.category || "civic_maintenance",
        department: currentClassification.department || currentExtractedData.department || "Municipal Administration",
        urgency: currentClassification.urgency || currentExtractedData.urgency || "medium",
        severity: currentClassification.severity || currentExtractedData.severity || "service_issue"
    };

    try {
        const response = await fetch('/api/grievance/confirm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        
        currentDraftToken = result.grievance_id;

        // Hide editable card, reveal success box
        const card = document.getElementById('draft-grievance-card');
        if (card) card.classList.add('hidden');

        const successBox = document.getElementById('draft-success-box');
        if (successBox) {
            successBox.classList.remove('hidden');
            const tokenEl = document.getElementById('confirmed-tracking-id');
            if (tokenEl) tokenEl.textContent = result.grievance_id;
        }

        appendChatMessage('assistant', `✅ **Grievance Confirmed & Registered!**\nYour official tracking ID is **${result.grievance_id}**.\nYou can track this grievance anytime in the **Track Complaint** tab.`);
        
        if (adminSessionPin) {
            loadAdminGrievances();
            loadAnalytics();
        }
        if (window.lucide) lucide.createIcons();
    } catch (err) {
        console.error(err);
        if (confirmBtn) {
            confirmBtn.disabled = false;
            confirmBtn.innerHTML = `<span>Confirm & Submit Complaint</span>`;
        }
    }
}

// Helper to track a specific grievance ID directly
function trackThisId(id) {
    const targetId = id || currentDraftToken;
    if (!targetId) return;
    const trackerInput = document.getElementById('tracker-input');
    if (trackerInput) {
        trackerInput.value = targetId;
    }
    switchTab('tracker');
    trackComplaint();
}

// RAG / FAQ Search
function runFaqExample(query) {
    const input = document.getElementById('faq-query');
    if (input) input.value = query;
    switchTab('faq');
    handleFaqSearch(new Event('submit'));
}

function fileGrievanceFromFaq() {
    const query = document.getElementById('faq-query')?.value || '';
    switchTab('complaint');
    if (query) {
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            chatInput.value = `I need to file a formal complaint regarding ${query}: `;
            chatInput.focus();
        }
    }
}

function renderFaqMarkdown(rawText) {
    if (!rawText) return '';
    let text = escapeHtml(rawText);

    // Convert ### Headings
    text = text.replace(/^###\s+(.+)$/gm, '<h3 class="text-xs sm:text-sm font-bold text-slate-900 mt-4 mb-2 flex items-center space-x-1.5 border-b border-slate-200/60 pb-1">$1</h3>');

    // Convert **bold**
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong class="font-bold text-slate-900">$1</strong>');

    // Convert bullet points • or - or *
    text = text.replace(/^[•\-\*]\s+(.+)$/gm, '<div class="flex items-start space-x-2 my-1"><span class="text-blue-600 font-bold select-none leading-none mt-1">•</span><span class="text-slate-800 text-xs sm:text-sm leading-relaxed">$1</span></div>');

    // Convert numbered lists
    text = text.replace(/^(\d+)\.\s+(.+)$/gm, '<div class="flex items-start space-x-2 my-1"><span class="w-4 h-4 rounded-full bg-slate-200 text-slate-700 flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">$1</span><span class="text-slate-800 text-xs sm:text-sm leading-relaxed">$2</span></div>');

    // Convert paragraphs
    const blocks = text.split(/\n\n+/);
    return blocks.map(b => {
        b = b.trim();
        if (b.startsWith('<h3') || b.startsWith('<div class="flex items-start')) {
            return b;
        }
        return `<p class="text-xs sm:text-sm text-slate-700 leading-relaxed">${b.replace(/\n/g, '<br/>')}</p>`;
    }).join('\n');
}

async function handleFaqSearch(event) {
    if (event) event.preventDefault();
    const queryInput = document.getElementById('faq-query');
    const query = queryInput ? queryInput.value.trim() : '';
    if (!query) {
        if (queryInput) queryInput.focus();
        return;
    }

    const searchBtn = document.getElementById('faq-search-btn');
    if (searchBtn) {
        searchBtn.disabled = true;
        searchBtn.innerHTML = `<i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i><span>Searching...</span>`;
        if (window.lucide) lucide.createIcons();
    }

    try {
        const res = await fetch('/api/rag/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await res.json();

        const answerBox = document.getElementById('faq-answer-box');
        const answerText = document.getElementById('faq-answer-text');
        const sourcesBox = document.getElementById('faq-sources-badges');

        if (answerBox) answerBox.classList.remove('hidden');
        if (answerText) {
            answerText.innerHTML = renderFaqMarkdown(data.answer);
        }

        if (sourcesBox) {
            sourcesBox.innerHTML = '';
            if (data.sources && data.sources.length > 0) {
                data.sources.forEach(src => {
                    const sp = document.createElement('span');
                    sp.className = 'inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-blue-50 text-blue-700 border border-blue-200 text-xs font-semibold font-mono shadow-2xs';
                    sp.innerHTML = `<i data-lucide="file-text" class="w-3.5 h-3.5 text-blue-500"></i><span>${escapeHtml(src)}</span>`;
                    sourcesBox.appendChild(sp);
                });
            }
        }
        if (window.lucide) lucide.createIcons();
    } catch (e) {
        console.error("FAQ search error:", e);
    } finally {
        if (searchBtn) {
            searchBtn.disabled = false;
            searchBtn.innerHTML = `<i data-lucide="sparkles" class="w-3.5 h-3.5"></i><span>Search</span>`;
            if (window.lucide) lucide.createIcons();
        }
    }
}

// Registry & Drawer View (Protected by Admin PIN)
async function loadRegistry() {
    if (adminSessionPin) {
        return loadAdminGrievances();
    }
}

async function viewGrievance(id) {
    if (!adminSessionPin) return;
    try {
        const res = await fetch(`/api/admin/grievances/${encodeURIComponent(id)}`, {
            headers: { 'X-Admin-PIN': adminSessionPin }
        });
        if (res.ok) {
            const data = await res.json();
            openAdminDrawer(data.grievance_id, data.summary || data.original_text, data.status, data.admin_remarks);
        }
    } catch (e) {
        console.error("View grievance error:", e);
    }
}

function closeModal() {
    closeAdminDrawer();
}

// Analytics Dashboard
async function loadAnalytics() {
    const deptSelect = document.getElementById('analytics-filter-dept');
    const dateSelect = document.getElementById('analytics-filter-date');
    const loadingBanner = document.getElementById('analytics-loading');
    const errorBanner = document.getElementById('analytics-error');
    const emptyBanner = document.getElementById('analytics-empty');
    const refreshIcon = document.getElementById('analytics-refresh-icon');

    const selectedDept = deptSelect ? deptSelect.value : 'all';
    const selectedPeriod = dateSelect ? dateSelect.value : 'all';

    let startDate = null;
    const now = new Date();
    if (selectedPeriod === '7d') {
        const d = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        startDate = d.toISOString().slice(0, 10);
    } else if (selectedPeriod === '30d') {
        const d = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        startDate = d.toISOString().slice(0, 10);
    }

    let url = '/api/analytics?';
    const params = new URLSearchParams();
    if (selectedDept && selectedDept !== 'all') {
        params.append('department', selectedDept);
    }
    if (startDate) {
        params.append('start_date', startDate);
    }
    url += params.toString();

    // UI state: loading
    if (loadingBanner) loadingBanner.classList.remove('hidden');
    if (errorBanner) errorBanner.classList.add('hidden');
    if (refreshIcon) refreshIcon.classList.add('animate-spin');

    try {
        const res = await fetch(url);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        const data = await res.json();

        // Check empty state
        if (data.total_registered === 0) {
            if (emptyBanner) emptyBanner.classList.remove('hidden');
        } else {
            if (emptyBanner) emptyBanner.classList.add('hidden');
        }

        // Update KPI Cards
        const kpiTotal = document.getElementById('kpi-total');
        const kpiSubmitted = document.getElementById('kpi-submitted');
        const kpiPending = document.getElementById('kpi-pending');
        const kpiInprogress = document.getElementById('kpi-inprogress');
        const kpiResolved = document.getElementById('kpi-resolved');
        const kpiResolution = document.getElementById('kpi-resolution-time');
        const kpiResolutionSub = document.getElementById('kpi-resolution-sub');

        if (kpiTotal) kpiTotal.textContent = data.total_registered ?? 0;
        if (kpiSubmitted) kpiSubmitted.textContent = data.total_submitted ?? 0;
        if (kpiPending) kpiPending.textContent = data.pending_action ?? 0;
        if (kpiInprogress) kpiInprogress.textContent = data.in_progress ?? 0;
        if (kpiResolved) kpiResolved.textContent = data.resolved_grievances ?? 0;
        
        if (kpiResolution) {
            kpiResolution.textContent = data.resolution_metrics?.avg_formatted || 'N/A';
        }
        if (kpiResolutionSub) {
            const countWithDur = data.resolution_metrics?.resolved_count_with_duration || 0;
            kpiResolutionSub.textContent = countWithDur > 0
                ? `Based on ${countWithDur} closed case${countWithDur > 1 ? 's' : ''}`
                : 'No closed cases yet';
        }

        // Update Header Active Count Badge
        const headerActive = document.getElementById('header-active-count');
        if (headerActive) {
            headerActive.textContent = (data.pending_action || 0) + (data.in_progress || 0);
        }

        // Update Last Updated Timestamp
        const lastUpdatedEl = document.getElementById('analytics-last-updated');
        if (lastUpdatedEl && data.last_updated) {
            const dateObj = new Date(data.last_updated);
            lastUpdatedEl.textContent = `Updated: ${dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
        }

        // Update ML Metrics Matrix
        if (data.ml_metrics && data.ml_metrics.tasks) {
            const t = data.ml_metrics.tasks;
            const formatMetricPct = v => (v == null ? '-' : ((v <= 1.0 ? v * 100 : v).toFixed(1) + '%'));
            if (t.intent) {
                const f1 = document.getElementById('ml-intent-f1');
                const acc = document.getElementById('ml-intent-acc');
                if (f1) f1.textContent = formatMetricPct(t.intent.weighted_f1);
                if (acc) acc.textContent = formatMetricPct(t.intent.accuracy);
            }
            if (t.department) {
                const f1 = document.getElementById('ml-dept-f1');
                const acc = document.getElementById('ml-dept-acc');
                if (f1) f1.textContent = formatMetricPct(t.department.weighted_f1);
                if (acc) acc.textContent = formatMetricPct(t.department.accuracy);
            }
            if (t.category) {
                const f1 = document.getElementById('ml-cat-f1');
                const acc = document.getElementById('ml-cat-acc');
                if (f1) f1.textContent = formatMetricPct(t.category.weighted_f1);
                if (acc) acc.textContent = formatMetricPct(t.category.accuracy);
            }
            if (t.urgency) {
                const f1 = document.getElementById('ml-urg-f1');
                const acc = document.getElementById('ml-urg-acc');
                if (f1) f1.textContent = formatMetricPct(t.urgency.weighted_f1);
                if (acc) acc.textContent = formatMetricPct(t.urgency.accuracy);
            }
        }

        // ─── 1. Chart: Department Workload ───
        const deptCanvas = document.getElementById('chart-departments');
        if (deptCanvas) {
            const deptLabels = (data.departments || []).map(d => d.label.replace(' Department', '').replace(' (TANGEDCO)', ''));
            const deptCounts = (data.departments || []).map(d => d.count);

            if (deptChart) deptChart.destroy();
            deptChart = new Chart(deptCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: deptLabels.length ? deptLabels : ['No Complaints'],
                    datasets: [{
                        label: 'Grievances',
                        data: deptCounts.length ? deptCounts : [0],
                        backgroundColor: '#0f172a',
                        borderRadius: 8,
                        borderSkipped: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } },
                        y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b', precision: 0 }, beginAtZero: true }
                    }
                }
            });
        }

        // ─── 2. Chart: Status Pipeline ───
        const statusCanvas = document.getElementById('chart-status');
        if (statusCanvas) {
            const sMap = data.by_status || {};
            const statusLabels = ['Confirmed', 'Under Review', 'In Progress', 'Forwarded', 'Resolved', 'Draft'];
            const statusCounts = [
                sMap.CONFIRMED || 0,
                sMap.UNDER_REVIEW || 0,
                sMap.IN_PROGRESS || 0,
                sMap.FORWARDED || 0,
                sMap.RESOLVED || 0,
                sMap.DRAFT || 0
            ];

            if (statusChart) statusChart.destroy();
            statusChart = new Chart(statusCanvas.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: statusLabels,
                    datasets: [{
                        data: statusCounts,
                        backgroundColor: ['#3b82f6', '#f59e0b', '#8b5cf6', '#f97316', '#10b981', '#94a3b8'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } }
                    }
                }
            });
        }

        // ─── 3. Chart: Category Breakdown ───
        const catCanvas = document.getElementById('chart-categories');
        if (catCanvas) {
            const catLabels = (data.categories || []).map(c => c.label.replace(/_/g, ' '));
            const catCounts = (data.categories || []).map(c => c.count);

            if (catChart) catChart.destroy();
            catChart = new Chart(catCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: catLabels.length ? catLabels : ['No Data'],
                    datasets: [{
                        label: 'Count',
                        data: catCounts.length ? catCounts : [0],
                        backgroundColor: '#8b5cf6',
                        borderRadius: 6,
                        borderSkipped: false
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: '#f1f5f9' }, ticks: { precision: 0 }, beginAtZero: true },
                        y: { grid: { display: false }, ticks: { font: { size: 10 } } }
                    }
                }
            });
        }

        // ─── 4. Chart: Urgency Levels ───
        const urgCanvas = document.getElementById('chart-urgency');
        if (urgCanvas) {
            const urgencies = data.urgencies || [];
            let high = 0, med = 0, low = 0;
            urgencies.forEach(u => {
                const l = (u.label || '').toLowerCase();
                if (l === 'high') high += u.count;
                else if (l === 'medium') med += u.count;
                else if (l === 'low') low += u.count;
            });

            if (urgChart) urgChart.destroy();
            urgChart = new Chart(urgCanvas.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: ['High', 'Medium', 'Low'],
                    datasets: [{
                        data: [high, med, low],
                        backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } }
                    }
                }
            });
        }

        // ─── 5. Chart: Trends Over Time ───
        const trendCanvas = document.getElementById('chart-trends');
        if (trendCanvas) {
            const trends = data.daily_submissions || [];
            const trendLabels = trends.map(t => t.date);
            const trendCounts = trends.map(t => t.count);

            if (trendChart) trendChart.destroy();
            trendChart = new Chart(trendCanvas.getContext('2d'), {
                type: 'line',
                data: {
                    labels: trendLabels.length ? trendLabels : ['Today'],
                    datasets: [{
                        label: 'New Grievances',
                        data: trendCounts.length ? trendCounts : [0],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointBackgroundColor: '#10b981',
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                        y: { grid: { color: '#f1f5f9' }, ticks: { precision: 0 }, beginAtZero: true }
                    }
                }
            });
        }

        // ─── 6. Chart: ML Benchmark Comparison ───
        const mlCanvas = document.getElementById('chart-ml-benchmark');
        if (mlCanvas) {
            const tasks = ['Intent', 'Department', 'Category', 'Urgency'];
            const t = data.ml_metrics?.tasks || {};
            const toNumPct = (v, fallback) => (v == null ? fallback : (v <= 1.0 ? Math.round(v * 1000) / 10 : Math.round(v * 10) / 10));
            const f1Scores = [
                toNumPct(t.intent?.weighted_f1, 97.8),
                toNumPct(t.department?.weighted_f1, 81.0),
                toNumPct(t.category?.weighted_f1, 74.6),
                toNumPct(t.urgency?.weighted_f1, 59.2)
            ];
            const accScores = [
                toNumPct(t.intent?.accuracy, 96.7),
                toNumPct(t.department?.accuracy, 83.3),
                toNumPct(t.category?.accuracy, 76.7),
                toNumPct(t.urgency?.accuracy, 58.3)
            ];

            if (mlBenchmarkChart) mlBenchmarkChart.destroy();
            mlBenchmarkChart = new Chart(mlCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: tasks,
                    datasets: [
                        {
                            label: 'Weighted F1 (%)',
                            data: f1Scores,
                            backgroundColor: '#8b5cf6',
                            borderRadius: 6,
                            borderSkipped: false
                        },
                        {
                            label: 'Accuracy (%)',
                            data: accScores,
                            backgroundColor: '#3b82f6',
                            borderRadius: 6,
                            borderSkipped: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } }
                    },
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                        y: {
                            min: 0,
                            max: 100,
                            grid: { color: '#f1f5f9' },
                            ticks: { callback: v => v + '%' }
                        }
                    }
                }
            });
        }

    } catch (e) {
        console.error("Analytics load error:", e);
        if (errorBanner) {
            const errText = document.getElementById('analytics-error-text');
            if (errText) errText.textContent = `Error loading analytics: ${e.message || 'Database unavailable'}`;
            errorBanner.classList.remove('hidden');
        }
    } finally {
        if (loadingBanner) loadingBanner.classList.add('hidden');
        if (refreshIcon) refreshIcon.classList.remove('animate-spin');
        if (window.lucide) lucide.createIcons();
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    checkSystemHealth();
    loadAnalytics();
});

// ─── CITIZEN COMPLAINT TRACKER ─────────────────────────────────────────────────

const STATUS_META = {
    DRAFT:        { label: 'Draft',        color: 'text-slate-700 font-semibold',   bg: 'bg-slate-100',        border: 'border-slate-250',   step: 0 },
    CONFIRMED:    { label: 'Confirmed',    color: 'text-blue-800 font-semibold',    bg: 'bg-blue-50',          border: 'border-blue-200',    step: 1 },
    UNDER_REVIEW: { label: 'Under Review', color: 'text-amber-800 font-semibold',   bg: 'bg-amber-50',         border: 'border-amber-200',   step: 2 },
    IN_PROGRESS:  { label: 'In Progress',  color: 'text-purple-800 font-semibold',  bg: 'bg-purple-50',        border: 'border-purple-200',  step: 3 },
    FORWARDED:    { label: 'Forwarded',    color: 'text-orange-800 font-semibold',  bg: 'bg-orange-50',        border: 'border-orange-200',  step: 3 },
    RESOLVED:     { label: 'Resolved',     color: 'text-emerald-800 font-semibold', bg: 'bg-emerald-50',       border: 'border-emerald-200', step: 4 },
};

const STEPS = ['Filed', 'Confirmed', 'Under Review', 'In Progress', 'Resolved'];

async function trackComplaint() {
    const rawInput = document.getElementById('tracker-input').value.trim();
    const grvId = rawInput.toUpperCase();
    if (!grvId) return;

    document.getElementById('tracker-error').classList.add('hidden');
    document.getElementById('tracker-result').classList.add('hidden');

    try {
        const res = await fetch(`/api/track/${encodeURIComponent(grvId)}`);
        if (!res.ok) {
            const err = await res.json();
            document.getElementById('tracker-error-text').textContent = err.detail || 'Grievance ID not found.';
            document.getElementById('tracker-error').classList.remove('hidden');
            return;
        }
        const d = await res.json();
        const meta = STATUS_META[d.status] || STATUS_META['DRAFT'];

        // Status banner
        const banner = document.getElementById('tracker-status-banner');
        banner.className = `p-4 rounded-xl border flex items-center justify-between ${meta.bg} ${meta.border}`;
        document.getElementById('tracker-grv-id').textContent = d.grievance_id;
        const badge = document.getElementById('tracker-status-badge');
        badge.textContent = meta.label;
        badge.className = `px-4 py-1.5 rounded-full text-sm font-bold border ${meta.color} ${meta.border} ${meta.bg}`;

        // Stepper
        renderStatusStepper(meta.step);

        // Details
        document.getElementById('tracker-category').textContent   = (d.category || '-').replace(/_/g, ' ');
        document.getElementById('tracker-department').textContent = d.department || '-';
        const urgency = d.urgency || 'medium';
        const urgEl = document.getElementById('tracker-urgency');
        urgEl.textContent = urgency.toUpperCase();
        urgEl.className = urgency === 'high' ? 'font-bold uppercase text-rose-400'
                        : urgency === 'low'  ? 'font-bold uppercase text-emerald-400'
                        : 'font-bold uppercase text-amber-400';
        document.getElementById('tracker-location').textContent  = d.location || 'Not specified';
        document.getElementById('tracker-created').textContent   = d.created_at ? d.created_at.slice(0,10) : '-';
        document.getElementById('tracker-updated').textContent   = d.updated_at ? d.updated_at.slice(0,10) : '-';

        // Official remarks
        const remarksCard = document.getElementById('tracker-remarks-card');
        if (d.admin_remarks && d.admin_remarks.trim()) {
            document.getElementById('tracker-remarks-text').textContent = d.admin_remarks;
            remarksCard.classList.remove('hidden');
        } else {
            remarksCard.classList.add('hidden');
        }

        document.getElementById('tracker-result').classList.remove('hidden');
        if (window.lucide) lucide.createIcons();
    } catch (e) {
        document.getElementById('tracker-error-text').textContent = 'Network error. Please try again.';
        document.getElementById('tracker-error').classList.remove('hidden');
        console.error('Tracker error:', e);
    }
}

function renderStatusStepper(activeStep) {
    const container = document.getElementById('tracker-stepper');
    container.innerHTML = '';
    STEPS.forEach((label, idx) => {
        const done    = idx < activeStep;
        const current = idx === activeStep;
        // Circle
        const circle = document.createElement('div');
        circle.className = `w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 border-2 ${
            done    ? 'bg-emerald-500 border-emerald-400 text-white' :
            current ? 'bg-blue-600 border-blue-400 text-white animate-pulse' :
                      'bg-slate-800 border-slate-600 text-slate-500'}`;
        circle.textContent = done ? '✓' : (idx + 1);
        // Label
        const labelEl = document.createElement('div');
        labelEl.className = `text-[10px] mt-1 text-center ${done ? 'text-emerald-400' : current ? 'text-blue-400 font-semibold' : 'text-slate-500'}`;
        labelEl.textContent = label;
        // Wrapper
        const wrap = document.createElement('div');
        wrap.className = 'flex flex-col items-center';
        wrap.appendChild(circle);
        wrap.appendChild(labelEl);
        container.appendChild(wrap);
        // Connector line (not after last)
        if (idx < STEPS.length - 1) {
            const line = document.createElement('div');
            line.className = `flex-1 h-0.5 mx-1 ${done ? 'bg-emerald-500' : 'bg-slate-700'}`;
            container.appendChild(line);
        }
    });
}

// Also allow Enter key in tracker input
document.addEventListener('DOMContentLoaded', () => {
    const trackerInput = document.getElementById('tracker-input');
    if (trackerInput) {
        trackerInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') trackComplaint();
        });
    }
});

// ─── ADMIN PORTAL ──────────────────────────────────────────────────────────────

async function adminLogin() {
    const pin = document.getElementById('admin-pin-input').value;
    const errEl = document.getElementById('admin-login-error');
    errEl.classList.add('hidden');

    try {
        // Verify PIN against the admin/stats endpoint
        const res = await fetch('/api/admin/stats', {
            headers: { 'X-Admin-PIN': pin }
        });
        if (res.status === 401) {
            errEl.classList.remove('hidden');
            return;
        }
        const stats = await res.json();
        adminSessionPin = pin;

        // Show dashboard, hide login card
        document.getElementById('admin-login-card').classList.add('hidden');
        document.getElementById('admin-dashboard').classList.remove('hidden');

        // Populate KPI cards
        document.getElementById('adm-total').textContent     = stats.total       || 0;
        document.getElementById('adm-confirmed').textContent = stats.confirmed    || 0;
        document.getElementById('adm-review').textContent    = stats.under_review || 0;
        document.getElementById('adm-inprogress').textContent= stats.in_progress  || 0;
        document.getElementById('adm-forwarded').textContent = stats.forwarded    || 0;
        document.getElementById('adm-resolved').textContent  = stats.resolved     || 0;

        await loadAdminGrievances();
        if (window.lucide) lucide.createIcons();
    } catch (e) {
        errEl.textContent = 'Server error. Please try again.';
        errEl.classList.remove('hidden');
        console.error('Admin login error:', e);
    }
}

function adminLogout() {
    adminSessionPin = null;
    currentAdminGrievanceId = null;
    document.getElementById('admin-dashboard').classList.add('hidden');
    document.getElementById('admin-login-card').classList.remove('hidden');
    document.getElementById('admin-pin-input').value = '';
    closeAdminDrawer();
}

async function loadAdminGrievances() {
    if (!adminSessionPin) return;

    const status = document.getElementById('adm-filter-status').value;
    const dept   = document.getElementById('adm-filter-dept').value;
    const search = document.getElementById('adm-search').value.trim();

    let url = `/api/admin/grievances?limit=100`;
    if (status) url += `&status=${encodeURIComponent(status)}`;
    if (dept)   url += `&department=${encodeURIComponent(dept)}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;

    try {
        const res = await fetch(url, { headers: { 'X-Admin-PIN': adminSessionPin } });
        if (res.status === 401) { adminLogout(); return; }
        const grievances = await res.json();

        const tbody = document.getElementById('admin-table-body');
        tbody.innerHTML = '';

        if (!grievances.length) {
            tbody.innerHTML = `<tr><td colspan="8" class="p-5 text-center text-slate-500 text-xs">No complaints found under this filter.</td></tr>`;
            return;
        }

        grievances.forEach(g => {
            const meta = STATUS_META[g.status] || STATUS_META['DRAFT'];
            const tr = document.createElement('tr');
            tr.className = 'hover:bg-slate-50/80 transition cursor-pointer border-b border-slate-100 text-xs';
            tr.innerHTML = `
                <td class="p-3 font-mono font-semibold text-slate-800 text-xs whitespace-nowrap">${escapeHtml(g.grievance_id)}</td>
                <td class="p-3 max-w-[200px] truncate text-slate-700 font-medium">${escapeHtml(g.summary || g.original_text?.slice(0,60) || '-')}</td>
                <td class="p-3 text-slate-800 font-medium whitespace-nowrap">${escapeHtml(g.department)}</td>
                <td class="p-3 capitalize font-semibold whitespace-nowrap ${g.urgency === 'high' ? 'text-rose-600' : g.urgency === 'low' ? 'text-emerald-600' : 'text-amber-600'}">${escapeHtml(g.urgency)}</td>
                <td class="p-3 text-slate-700 font-normal max-w-[180px] truncate">${escapeHtml(g.location || '-')}</td>
                <td class="p-3 whitespace-nowrap">
                    <span class="inline-block px-2.5 py-0.5 rounded-full border text-[11px] font-semibold whitespace-nowrap ${meta.color} ${meta.border} ${meta.bg}">
                        ${meta.label}
                    </span>
                </td>
                <td class="p-3 text-slate-600 whitespace-nowrap">${(g.created_at || '').slice(0,10)}</td>
                <td class="p-3 text-center whitespace-nowrap">
                    <button onclick="openAdminDrawer('${escapeHtml(g.grievance_id)}', '${escapeHtml(g.summary || '')}', '${escapeHtml(g.status)}', \`${escapeHtml(g.admin_remarks || '')}\`)"
                            class="px-3 py-1 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-medium transition shadow-xs">
                        Update
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        if (window.lucide) lucide.createIcons();
    } catch (e) {
        console.error('Admin load error:', e);
    }
}

function openAdminDrawer(grvId, summary, currentStatus, currentRemarks) {
    currentAdminGrievanceId = grvId;
    document.getElementById('drawer-grv-id').textContent = grvId;
    document.getElementById('drawer-summary').textContent = summary || '(no summary)';
    document.getElementById('drawer-status').value = currentStatus || 'CONFIRMED';
    document.getElementById('drawer-remarks').value = currentRemarks || '';
    document.getElementById('drawer-toast').classList.add('hidden');

    const drawer = document.getElementById('admin-drawer');
    drawer.classList.remove('hidden');
    drawer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    if (window.lucide) lucide.createIcons();
}

function closeAdminDrawer() {
    document.getElementById('admin-drawer').classList.add('hidden');
    currentAdminGrievanceId = null;
}

async function submitAdminUpdate() {
    if (!currentAdminGrievanceId || !adminSessionPin) return;
    const newStatus  = document.getElementById('drawer-status').value;
    const remarks    = document.getElementById('drawer-remarks').value.trim();
    const btn        = document.getElementById('drawer-submit-btn');
    const toast      = document.getElementById('drawer-toast');

    btn.disabled = true;
    btn.innerHTML = `<span>Saving...</span>`;

    try {
        const res = await fetch(`/api/admin/grievances/${encodeURIComponent(currentAdminGrievanceId)}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json',
                'X-Admin-PIN': adminSessionPin
            },
            body: JSON.stringify({ status: newStatus, admin_remarks: remarks })
        });

        if (res.ok) {
            toast.textContent = `✅ Updated successfully — Status: ${newStatus}`;
            toast.className = 'text-xs text-center py-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
            toast.classList.remove('hidden');
            // Refresh table, analytics, and KPI stats
            await loadAdminGrievances();
            loadAnalytics();
            const statsRes = await fetch('/api/admin/stats', { headers: { 'X-Admin-PIN': adminSessionPin } });
            if (statsRes.ok) {
                const stats = await statsRes.json();
                document.getElementById('adm-total').textContent      = stats.total       || 0;
                document.getElementById('adm-confirmed').textContent  = stats.confirmed    || 0;
                document.getElementById('adm-review').textContent     = stats.under_review || 0;
                document.getElementById('adm-inprogress').textContent = stats.in_progress  || 0;
                document.getElementById('adm-forwarded').textContent  = stats.forwarded    || 0;
                document.getElementById('adm-resolved').textContent   = stats.resolved     || 0;
            }
            setTimeout(() => toast.classList.add('hidden'), 3000);
        } else {
            const err = await res.json();
            toast.textContent = `❌ Error: ${err.detail || 'Update failed'}`;
            toast.className = 'text-xs text-center py-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20';
            toast.classList.remove('hidden');
        }
    } catch (e) {
        toast.textContent = '❌ Network error. Please try again.';
        toast.className = 'text-xs text-center py-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20';
        toast.classList.remove('hidden');
        console.error('Admin update error:', e);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="save" class="w-4 h-4"></i><span>Save Update</span>`;
        if (window.lucide) lucide.createIcons();
    }
}
