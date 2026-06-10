/* app.js — Interactivity and API integration for LexRAG */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadProgressContainer = document.getElementById('uploadProgressContainer');
    const uploadFilename = document.getElementById('uploadFilename');
    const uploadProgressPct = document.getElementById('uploadProgressPct');
    const uploadProgressBarFill = document.getElementById('uploadProgressBarFill');
    const uploadStatusText = document.getElementById('uploadStatusText');

    const statDocs = document.getElementById('statDocs');
    const statPages = document.getElementById('statPages');
    const statCache = document.getElementById('statCache');
    const statChunks = document.getElementById('statChunks');
    const activeDocsUl = document.getElementById('activeDocsUl');

    const queryForm = document.getElementById('queryForm');
    const queryInput = document.getElementById('queryInput');
    const submitBtn = document.getElementById('submitBtn');
    
    const loadingShimmer = document.getElementById('loadingShimmer');
    const comparisonOutput = document.getElementById('comparisonOutput');

    const lexragLatency = document.getElementById('lexragLatency');
    const lexragChunks = document.getElementById('lexragChunks');
    const lexragAnswer = document.getElementById('lexragAnswer');
    const lexragConfidence = document.getElementById('lexragConfidence');
    const lexragCitations = document.getElementById('lexragCitations');

    const baselineLatency = document.getElementById('baselineLatency');
    const baselineChunks = document.getElementById('baselineChunks');
    const baselineAnswer = document.getElementById('baselineAnswer');
    const baselineConfidence = document.getElementById('baselineConfidence');
    const baselineCitations = document.getElementById('baselineCitations');

    const analysisReasons = document.getElementById('analysisReasons');

    // Load Initial Stats
    fetchStats();

    // Setup Sample Query Buttons
    document.querySelectorAll('.sample-badge').forEach(badge => {
        badge.addEventListener('click', () => {
            queryInput.value = badge.textContent;
            queryForm.dispatchEvent(new Event('submit'));
        });
    });

    // ==========================================================================
    // Corpus Stats Fetching & Rendering
    // ==========================================================================
    function fetchStats() {
        fetch('/api/stats')
            .then(res => {
                if (!res.ok) throw new Error("Stats load error");
                return res.json();
            })
            .then(data => {
                // Update Stat Cards
                statDocs.textContent = data.num_docs;
                statPages.textContent = data.total_pages;
                statCache.textContent = data.cache_size;
                statChunks.textContent = `${data.baseline_chunks} / ${data.lexrag_subchunks}`;

                // Update Active Document List
                activeDocsUl.innerHTML = '';
                if (data.documents && data.documents.length > 0) {
                    data.documents.forEach(doc => {
                        const li = document.createElement('li');
                        li.innerHTML = `
                            <span class="doc-title" title="${doc.title}">${doc.title}</span>
                            <span class="doc-meta">
                                <span>Domain: ${doc.domain}</span>
                                <span>Pages: ${doc.num_pages}</span>
                            </span>
                        `;
                        activeDocsUl.appendChild(li);
                    });
                } else {
                    activeDocsUl.innerHTML = '<li class="no-docs">No documents ingested yet. Upload a PDF above!</li>';
                }
            })
            .catch(err => {
                console.error("Error loading stats:", err);
                activeDocsUl.innerHTML = '<li class="error-doc">Error loading stats. Verify backend is running.</li>';
            });
    }

    // ==========================================================================
    // File Ingestion Upload Handlers
    // ==========================================================================
    
    // Clicking drop zone opens file dialogue
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag-over styling
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    // File dropped
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // File input changes manually
    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.name.endsWith('.pdf')) {
            alert('Error: Please upload a valid PDF file.');
            return;
        }

        // Show Progress Container
        uploadProgressContainer.classList.remove('hidden');
        uploadFilename.textContent = file.name;
        uploadProgressPct.textContent = '0%';
        uploadProgressBarFill.style.width = '0%';
        uploadStatusText.textContent = 'Uploading PDF...';
        
        // Lock Query Input during upload and re-indexing
        submitBtn.disabled = true;
        queryInput.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        // Upload using XMLHttpRequest to monitor progress
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/ingest', true);
        const _ikey = getStoredKey();
        if (_ikey) xhr.setRequestHeader('X-OpenAI-Api-Key', _ikey);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                // Cap progress indicator at 90% until server responds to re-indexing
                const displayPct = Math.round(pct * 0.9);
                uploadProgressPct.textContent = `${displayPct}%`;
                uploadProgressBarFill.style.width = `${displayPct}%`;
                if (pct === 100) {
                    uploadStatusText.textContent = 'PDF Upload complete. Chunking and rebuilding FAISS indices (checking cache)...';
                }
            }
        };

        xhr.onload = () => {
            submitBtn.disabled = false;
            queryInput.disabled = false;

            if (xhr.status === 200) {
                const resp = JSON.parse(xhr.responseText);
                uploadProgressPct.textContent = '100%';
                uploadProgressBarFill.style.width = '100%';
                uploadStatusText.textContent = 'Indexing Complete! System ready.';
                
                // Color progress bar green
                uploadProgressBarFill.style.background = 'var(--color-success)';
                uploadProgressBarFill.style.boxShadow = '0 0 8px var(--color-success)';
                
                // Refresh Corpus Stats
                fetchStats();
                
                setTimeout(() => {
                    uploadProgressContainer.classList.add('hidden');
                }, 3000);
            } else {
                const errorResp = JSON.parse(xhr.responseText || '{"detail": "Unknown server error during re-indexing"}');
                uploadStatusText.textContent = `Processing failed: ${errorResp.detail}`;
                uploadProgressBarFill.style.background = 'var(--color-danger)';
                uploadProgressBarFill.style.boxShadow = '0 0 8px var(--color-danger)';
            }
        };

        xhr.onerror = () => {
            submitBtn.disabled = false;
            queryInput.disabled = false;
            uploadStatusText.textContent = 'Network upload error. Check backend connection.';
            uploadProgressBarFill.style.background = 'var(--color-danger)';
        };

        xhr.send(formData);
    }

    // ==========================================================================
    // Query Submit Comparative Handler
    // ==========================================================================
    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const queryText = queryInput.value.trim();
        if (!queryText) return;

        // Visual layout state updates
        submitBtn.disabled = true;
        queryInput.disabled = true;
        comparisonOutput.classList.add('hidden');
        loadingShimmer.classList.remove('hidden');

        const _qkey = getStoredKey();
        fetch('/api/query', {
            method: 'POST',
            headers: Object.assign(
                { 'Content-Type': 'application/json' },
                _qkey ? { 'X-OpenAI-Api-Key': _qkey } : {}
            ),
            body: JSON.stringify({ query: queryText })
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => {
                    throw new Error(err.detail || "Query execution failed.");
                });
            }
            return res.json();
        })
        .then(data => {
            // Populate LexRAG Card
            lexragLatency.textContent = data.lexrag.latency_ms;
            lexragChunks.textContent = data.lexrag.chunks_used;
            lexragAnswer.innerHTML = formatAnswerText(data.lexrag.answer);
            
            // Confidence Badge Styling
            lexragConfidence.textContent = data.lexrag.confidence.toUpperCase();
            if (data.lexrag.confidence.toLowerCase() === 'high') {
                lexragConfidence.className = 'badge confidence-badge high';
            } else {
                lexragConfidence.className = 'badge confidence-badge low';
            }

            // Populate LexRAG Citations
            lexragCitations.innerHTML = '';
            if (data.lexrag.citations && data.lexrag.citations.length > 0) {
                data.lexrag.citations.forEach(cit => {
                    const div = document.createElement('div');
                    div.className = 'citation-card';
                    div.innerHTML = `
                        <div class="cit-num">${cit.source_num}</div>
                        <div class="cit-details">
                            <span class="cit-file" title="${cit.doc_id}">${cit.doc_id}</span>
                            <span class="cit-page">Page: ${cit.page_number}</span>
                        </div>
                    `;
                    lexragCitations.appendChild(div);
                });
            } else {
                lexragCitations.innerHTML = '<div class="no-citations">No citations referenced in response.</div>';
            }

            // Populate Baseline RAG Card
            baselineLatency.textContent = data.baseline.latency_ms;
            baselineChunks.textContent = data.baseline.chunks_used;
            baselineAnswer.innerHTML = formatAnswerText(data.baseline.answer);
            
            baselineConfidence.textContent = data.baseline.confidence.toUpperCase();
            if (data.baseline.confidence.toLowerCase() === 'high') {
                baselineConfidence.className = 'badge confidence-badge high';
            } else {
                baselineConfidence.className = 'badge confidence-badge low';
            }

            // Populate Baseline Citations
            baselineCitations.innerHTML = '';
            if (data.baseline.citations && data.baseline.citations.length > 0) {
                data.baseline.citations.forEach(cit => {
                    const div = document.createElement('div');
                    // Check if page boundaries were blown (-1)
                    const isBlown = (cit.page_number === -1 || cit.page_number === "-1");
                    div.className = `citation-card ${isBlown ? 'boundary-alert' : ''}`;
                    
                    div.innerHTML = `
                        <div class="cit-num">${cit.source_num}</div>
                        <div class="cit-details">
                            <span class="cit-file" title="${cit.doc_id}">${cit.doc_id}</span>
                            <span class="cit-page">${isBlown ? '🚨 Unknown Page (-1) - Boundary Blown!' : `Page: ${cit.page_number}`}</span>
                        </div>
                    `;
                    baselineCitations.appendChild(div);
                });
            } else {
                baselineCitations.innerHTML = '<div class="no-citations">No citations referenced in response.</div>';
            }

            // Populate Analysis Reasons
            analysisReasons.innerHTML = '';
            if (data.comparison && data.comparison.reasons) {
                data.comparison.reasons.forEach(reason => {
                    const li = document.createElement('li');
                    li.textContent = reason;
                    analysisReasons.appendChild(li);
                });
            }

            // Toggle view panels
            loadingShimmer.classList.add('hidden');
            comparisonOutput.classList.remove('hidden');
        })
        .catch(err => {
            alert(`Query failed: ${err.message}`);
            console.error("Query Error:", err);
            loadingShimmer.classList.add('hidden');
        })
        .finally(() => {
            submitBtn.disabled = false;
            queryInput.disabled = false;
        });
    });

    // =========================================================================
    // API Key Settings Modal
    // =========================================================================
    const STORAGE_KEY = 'lexrag_openai_key';
    const settingsBtn   = document.getElementById('settingsBtn');
    const settingsModal = document.getElementById('settingsModal');
    const settingsClose = document.getElementById('settingsClose');
    const apiKeyInput   = document.getElementById('apiKeyInput');
    const saveKeyBtn    = document.getElementById('saveKeyBtn');
    const clearKeyBtn   = document.getElementById('clearKeyBtn');
    const modeBadge     = document.getElementById('modeBadge');
    const providerStatus = document.getElementById('providerStatus');

    function getStoredKey() { return localStorage.getItem(STORAGE_KEY) || ''; }

    function updateBadge(key) {
        if (key) {
            modeBadge.textContent = 'OpenAI (GPT-4o-mini)';
            modeBadge.className = 'mode-badge openai';
            if (providerStatus) providerStatus.textContent = 'OpenAI Mode';
        } else {
            modeBadge.textContent = 'Open-Source (Free)';
            modeBadge.className = 'mode-badge oss';
            if (providerStatus) providerStatus.textContent = 'Open-Source Mode';
        }
    }
    updateBadge(getStoredKey());

    if (settingsBtn) {
        settingsBtn.addEventListener('click', () => {
            apiKeyInput.value = getStoredKey();
            updateBadge(getStoredKey());
            settingsModal.classList.add('open');
        });
    }
    if (settingsClose) {
        settingsClose.addEventListener('click', () => settingsModal.classList.remove('open'));
    }
    if (settingsModal) {
        settingsModal.addEventListener('click', e => {
            if (e.target === settingsModal) settingsModal.classList.remove('open');
        });
    }
    if (saveKeyBtn) {
        saveKeyBtn.addEventListener('click', () => {
            const key = apiKeyInput.value.trim();
            key ? localStorage.setItem(STORAGE_KEY, key) : localStorage.removeItem(STORAGE_KEY);
            updateBadge(key);
            settingsModal.classList.remove('open');
        });
    }
    if (clearKeyBtn) {
        clearKeyBtn.addEventListener('click', () => {
            localStorage.removeItem(STORAGE_KEY);
            apiKeyInput.value = '';
            updateBadge('');
        });
    }

    /**
     * Helper to wrap [SOURCE X] text snippets into custom visual elements
     */
    function formatAnswerText(text) {
        if (!text) return '';
        // Safe HTML Escape
        let escaped = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Format [SOURCE X] notation as interactive glowing badges
        return escaped.replace(/\[SOURCE (\d+)\]/g, (match, num) => {
            return `<span class="cit-tag" title="Click to view Source ${num}">[Source ${num}]</span>`;
        });
    }
});
