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
let urgencyChart = null;
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

    if (tabId === 'registry') loadRegistry();
    if (tabId === 'analytics') loadAnalytics();
    if (tabId === 'admin' && adminSessionPin) loadAdminGrievances();
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
    container.innerHTML = `
        <div class="flex items-start space-x-3">
            <div class="w-9 h-9 rounded-2xl bg-slate-900 flex items-center justify-center flex-shrink-0 text-white shadow-sm">
                <i data-lucide="bot" class="w-4 h-4"></i>
            </div>
            <div class="chat-bubble-ai p-4 max-w-[85%] text-sm leading-relaxed" style="color: #0f172a !important; background-color: #f8fafc !important; border: 1.5px solid #cbd5e1 !important;">
                <p class="font-bold text-slate-900 mb-1" style="color: #0f172a !important; -webkit-text-fill-color: #0f172a !important;">Session Reset. Welcome to GrievanceGPT.</p>
                <p class="text-slate-900 font-medium" style="color: #0f172a !important; -webkit-text-fill-color: #0f172a !important;">
                    Please describe your public grievance in English, தமிழ், or Tanglish.
                </p>
            </div>
        </div>
    `;

    updateCompletenessUI(0, []);
    document.getElementById('res-intent').textContent = '-';
    document.getElementById('res-category').textContent = '-';
    document.getElementById('res-department').textContent = '-';
    document.getElementById('res-urgency').textContent = '-';
    document.getElementById('draft-grievance-card').classList.add('hidden');
    document.getElementById('draft-placeholder-card').classList.remove('hidden');
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

        // Update Classification Panel
        if (data.classification) {
            document.getElementById('res-intent').textContent = data.classification.intent || '-';
            document.getElementById('res-category').textContent = (data.classification.category || '-').replace('_', ' ');
            document.getElementById('res-department').textContent = data.classification.department || '-';
            document.getElementById('res-urgency').textContent = (data.classification.urgency || '-').toUpperCase();
        }

        // Update Completeness
        if (data.completeness) {
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
            <div class="chat-bubble-ai p-4 max-w-[85%] text-sm leading-relaxed shadow-xs" style="color: #0f172a !important; background-color: #f8fafc !important; border: 1.5px solid #cbd5e1 !important;">
                <p class="whitespace-pre-line text-slate-900 font-medium" style="color: #0f172a !important; -webkit-text-fill-color: #0f172a !important; opacity: 1 !important;">${formatAssistantText(text)}</p>
            </div>
        `;
    }

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    if (window.lucide) lucide.createIcons();
}

function formatAssistantText(text) {
    return text.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #020617 !important; -webkit-text-fill-color: #020617 !important; font-weight: 700 !important;">$1</strong>');
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

    bar.style.width = score + '%';
    badge.textContent = score + '%';

    if (score < 40) {
        bar.className = 'h-2.5 rounded-full progress-fill bg-rose-500';
    } else if (score < 70) {
        bar.className = 'h-2.5 rounded-full progress-fill bg-amber-500';
    } else {
        bar.className = 'h-2.5 rounded-full progress-fill bg-emerald-500';
    }

    if (missingFields && missingFields.length > 0) {
        missingBox.classList.remove('hidden');
        missingText.textContent = missingFields.join(', ');
    } else {
        missingBox.classList.add('hidden');
    }
}

function populateDraftCard(draft) {
    document.getElementById('draft-placeholder-card').classList.add('hidden');
    const card = document.getElementById('draft-grievance-card');
    card.classList.remove('hidden');

    document.getElementById('draft-token').textContent = draft.grievance_id;
    document.getElementById('draft-subject').value = draft.subject;
    document.getElementById('draft-description').value = draft.description;
    document.getElementById('draft-location').value = draft.location;
    document.getElementById('draft-duration').value = draft.duration;
    currentDraftToken = draft.grievance_id;
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
    confirmBtn.disabled = true;
    confirmBtn.innerHTML = `<span>Saving...</span>`;

    const payload = {
        grievance_id: currentDraftToken,
        subject: document.getElementById('draft-subject').value,
        description: document.getElementById('draft-description').value,
        location: document.getElementById('draft-location').value,
        duration: document.getElementById('draft-duration').value,
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
        
        confirmBtn.className = 'flex-1 py-2 rounded-xl bg-slate-800 text-emerald-400 text-xs font-semibold flex items-center justify-center space-x-1 border border-emerald-500/30';
        confirmBtn.innerHTML = `<i data-lucide="check-check" class="w-4 h-4"></i><span>Confirmed (${result.grievance_id})</span>`;
        
        appendChatMessage('assistant', `✅ **Grievance Confirmed!**\nYour reference ID is **${result.grievance_id}**. The complaint is saved in the registry under status **CONFIRMED**.`);
        if (window.lucide) lucide.createIcons();
        loadRegistry();
        loadAnalytics();
    } catch (err) {
        console.error(err);
        confirmBtn.disabled = false;
        confirmBtn.innerHTML = `<span>Confirm Grievance</span>`;
    }
}

// Standalone Completeness Auditor
async function auditCompleteness() {
    const text = document.getElementById('reviewer-input').value.trim();
    if (!text) return;

    try {
        const res = await fetch('/api/completeness/audit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await res.json();

        document.getElementById('reviewer-results').classList.remove('hidden');
        document.getElementById('rev-score').textContent = data.score + '%';

        const badge = document.getElementById('rev-badge');
        if (data.is_complete) {
            badge.textContent = "Ready for Submission";
            badge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
        } else {
            badge.textContent = "Incomplete - Missing Details";
            badge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30";
        }

        const checklist = document.getElementById('rev-checklist');
        checklist.innerHTML = '';
        for (const [k, v] of Object.entries(data.field_status || {})) {
            const li = document.createElement('li');
            li.className = "flex items-center space-x-2";
            li.innerHTML = v 
                ? `<span class="text-emerald-400 font-bold">✓</span> <span class="capitalize">${k.replace('_', ' ')} detected</span>`
                : `<span class="text-rose-400 font-bold">✗</span> <span class="capitalize text-slate-400">Missing ${k.replace('_', ' ')}</span>`;
            checklist.appendChild(li);
        }

        const suggestions = document.getElementById('rev-suggestions');
        suggestions.innerHTML = '';
        if (data.suggestions && data.suggestions.length > 0) {
            data.suggestions.forEach(s => {
                const li = document.createElement('li');
                li.textContent = "• " + s;
                suggestions.appendChild(li);
            });
        } else {
            suggestions.innerHTML = '<li class="text-emerald-400">Excellent! All essential details are present.</li>';
        }

    } catch (e) {
        console.error("Audit error:", e);
    }
}

// RAG / FAQ Search
function runFaqExample(query) {
    document.getElementById('faq-query').value = query;
    handleFaqSearch(new Event('submit'));
}

async function handleFaqSearch(event) {
    if (event) event.preventDefault();
    const query = document.getElementById('faq-query').value.trim();
    if (!query) return;

    try {
        const res = await fetch('/api/rag/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await res.json();

        document.getElementById('faq-answer-box').classList.remove('hidden');
        document.getElementById('faq-answer-text').textContent = data.answer;

        const sourcesBox = document.getElementById('faq-sources-badges');
        sourcesBox.innerHTML = '';
        if (data.sources) {
            data.sources.forEach(src => {
                const sp = document.createElement('span');
                sp.className = 'px-2 py-0.5 rounded bg-purple-900/40 text-purple-300 border border-purple-700/50 text-[10px] font-mono';
                sp.textContent = src;
                sourcesBox.appendChild(sp);
            });
        }
    } catch (e) {
        console.error("FAQ search error:", e);
    }
}

// Registry Table
async function loadRegistry() {
    const status = document.getElementById('filter-status').value;
    const dept = document.getElementById('filter-dept').value;

    try {
        const res = await fetch(`/api/grievances?status=${encodeURIComponent(status)}&department=${encodeURIComponent(dept)}`);
        const grievances = await res.json();

        const tbody = document.getElementById('registry-table-body');
        tbody.innerHTML = '';

        if (grievances.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="p-6 text-center text-slate-600 font-medium">No grievances recorded under this filter.</td></tr>`;
            return;
        }

        grievances.forEach(g => {
            const tr = document.createElement('tr');
            tr.className = "hover:bg-slate-50/80 transition border-b border-slate-100 text-xs";
            const badgeClass = (g.status === 'CONFIRMED')
                ? 'bg-blue-50 text-blue-800 border-blue-200' 
                : (g.status === 'RESOLVED')
                ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                : (g.status === 'UNDER_REVIEW')
                ? 'bg-amber-50 text-amber-800 border-amber-200'
                : 'bg-slate-100 text-slate-700 border-slate-200';

            const statusLabel = g.status === 'UNDER_REVIEW' ? 'Under Review' : g.status === 'IN_PROGRESS' ? 'In Progress' : g.status;

            tr.innerHTML = `
                <td class="p-3 font-mono font-semibold text-slate-800 text-xs whitespace-nowrap">${g.grievance_id}</td>
                <td class="p-3 font-medium text-slate-800 whitespace-nowrap">${escapeHtml(g.department)}</td>
                <td class="p-3 font-medium text-slate-700 capitalize whitespace-nowrap">${escapeHtml(g.category.replace('_', ' '))}</td>
                <td class="p-3 text-slate-700 font-normal max-w-[200px] truncate">${escapeHtml(g.location || 'Not specified')}</td>
                <td class="p-3 font-semibold capitalize whitespace-nowrap ${g.urgency === 'high' ? 'text-rose-600' : g.urgency === 'low' ? 'text-emerald-600' : 'text-amber-600'}">${escapeHtml(g.urgency)}</td>
                <td class="p-3 whitespace-nowrap">
                    <span class="inline-block px-2.5 py-0.5 rounded-full border text-[11px] font-semibold whitespace-nowrap ${badgeClass}">
                        ${statusLabel}
                    </span>
                </td>
                <td class="p-3 text-right whitespace-nowrap">
                    <button onclick="viewGrievance('${g.grievance_id}')" class="px-3 py-1 bg-slate-900 hover:bg-slate-800 text-white font-medium rounded-lg text-xs transition shadow-xs">
                        View
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        if (window.lucide) lucide.createIcons();
    } catch (e) {
        console.error("Registry load error:", e);
    }
}

// Modal View
async function viewGrievance(id) {
    try {
        const res = await fetch(`/api/grievances/${id}`);
        const data = await res.json();

        document.getElementById('modal-title').textContent = `${data.grievance_id} - ${data.department}`;
        const modalBody = document.getElementById('modal-body');
        modalBody.innerHTML = `
            <div class="p-4 bg-slate-50 rounded-2xl border border-slate-200 space-y-2 font-mono text-xs whitespace-pre-wrap leading-relaxed text-slate-900 font-medium">
${escapeHtml(data.generated_grievance || data.summary)}
            </div>
            <div class="grid grid-cols-2 gap-3 text-xs pt-3">
                <div class="p-2.5 bg-slate-50 rounded-xl border border-slate-100"><span class="text-slate-500 font-medium">Location:</span> <span class="text-slate-900 font-bold">${escapeHtml(data.location || 'N/A')}</span></div>
                <div class="p-2.5 bg-slate-50 rounded-xl border border-slate-100"><span class="text-slate-500 font-medium">Duration:</span> <span class="text-slate-900 font-bold">${escapeHtml(data.duration || 'N/A')}</span></div>
                <div class="p-2.5 bg-slate-50 rounded-xl border border-slate-100"><span class="text-slate-500 font-medium">Urgency:</span> <span class="text-slate-900 font-bold">${escapeHtml(data.urgency)}</span></div>
                <div class="p-2.5 bg-slate-50 rounded-xl border border-slate-100"><span class="text-slate-500 font-medium">Status:</span> <span class="text-slate-900 font-bold">${escapeHtml(data.status)}</span></div>
            </div>
        `;
        document.getElementById('grievance-modal').classList.remove('hidden');
    } catch (e) {
        console.error(e);
    }
}

function closeModal() {
    document.getElementById('grievance-modal').classList.add('hidden');
}

// Analytics Dashboard
async function loadAnalytics() {
    try {
        const res = await fetch('/api/analytics');
        const data = await res.json();

        document.getElementById('kpi-total').textContent = data.total_grievances;
        document.getElementById('kpi-confirmed').textContent = data.confirmed_grievances;
        document.getElementById('kpi-drafts').textContent = data.draft_grievances;

        // Render Department Chart
        const deptLabels = data.departments.map(d => d.label);
        const deptCounts = data.departments.map(d => d.count);

        if (deptChart) deptChart.destroy();
        const ctxDept = document.getElementById('chart-departments').getContext('2d');
        deptChart = new Chart(ctxDept, {
            type: 'bar',
            data: {
                labels: deptLabels.length ? deptLabels : ['No Data Recorded'],
                datasets: [{
                    label: 'Complaints',
                    data: deptCounts.length ? deptCounts : [0],
                    backgroundColor: '#0f172a',
                    borderRadius: 10,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } },
                    y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b', precision: 0 } }
                }
            }
        });

        // Render Urgency Chart
        const urgLabels = data.urgencies.map(u => u.label.toUpperCase());
        const urgCounts = data.urgencies.map(u => u.count);

        if (urgChart) urgChart.destroy();
        const ctxUrg = document.getElementById('chart-urgency').getContext('2d');
        urgChart = new Chart(ctxUrg, {
            type: 'doughnut',
            data: {
                labels: urgLabels.length ? urgLabels : ['NO DATA'],
                datasets: [{
                    data: urgCounts.length ? urgCounts : [0],
                    backgroundColor: ['#f87171', '#fbbf24', '#3b82f6', '#10b981'],
                    borderWidth: 3,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#475569', font: { size: 11, weight: 'bold' } } }
                }
            }
        });

    } catch (e) {
        console.error("Analytics load error:", e);
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    checkSystemHealth();
    loadAnalytics();
    loadRegistry();
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
            // Refresh table and KPI stats
            await loadAdminGrievances();
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
