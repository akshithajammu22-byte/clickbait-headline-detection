document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let globalMetrics = null;
  let batchData = [];
  let sampleHeadlinesList = [];
  let datasetRows = [];

  const CIRCLE_CIRCUMFERENCE = 339.292; // 2 * PI * 54

  // Initialize Elements
  initNavigation();
  initLiveAnalyzer();
  initBatchProcessor();
  initPresetChips();
  fetchInitialData();

  // Navigation Tab Switching
  function initNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    const panes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.getAttribute('data-tab');
        tabs.forEach(t => t.classList.remove('active'));
        panes.forEach(p => p.classList.remove('active'));

        tab.classList.add('active');
        const targetPane = document.getElementById(`pane-${target}`);
        if (targetPane) targetPane.classList.add('active');

        // Load tab data on demand
        if (target === 'models' && !globalMetrics) loadModelBenchmarks();
        if (target === 'dataset' && datasetRows.length === 0) loadDatasetExplorer();
      });
    });
  }

  // Live Analyzer Logic
  function initLiveAnalyzer() {
    const input = document.getElementById('headline-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const sampleBtn = document.getElementById('random-sample-btn');

    analyzeBtn.addEventListener('click', () => runLivePrediction());
    clearBtn.addEventListener('click', () => {
      input.value = '';
      input.focus();
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        runLivePrediction();
      }
    });

    sampleBtn.addEventListener('click', () => {
      if (sampleHeadlinesList.length > 0) {
        const randomItem = sampleHeadlinesList[Math.floor(Math.random() * sampleHeadlinesList.length)];
        input.value = randomItem.text;
        runLivePrediction();
      } else {
        input.value = "10 Shocking Secrets Flight Attendants Never Tell Passengers! (#3 Is Insane)";
        runLivePrediction();
      }
    });
  }

  // Preset Chips
  function initPresetChips() {
    const chips = document.querySelectorAll('.chip');
    const input = document.getElementById('headline-input');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        input.value = chip.getAttribute('data-text');
        runLivePrediction();
      });
    });
  }

  // Execute Live Prediction
  async function runLivePrediction() {
    const input = document.getElementById('headline-input');
    const headline = input.value.trim();
    if (!headline) {
      alert('Please enter a headline text to analyze.');
      input.focus();
      return;
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    const origBtnHtml = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = `Analyzing...`;
    analyzeBtn.disabled = true;

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ headline })
      });
      const data = await resp.json();

      if (data.error) {
        alert(data.error);
        return;
      }
      renderLiveResult(data);
    } catch (err) {
      console.error('Prediction failed:', err);
      alert('Failed to connect to the prediction server.');
    } finally {
      analyzeBtn.innerHTML = origBtnHtml;
      analyzeBtn.disabled = false;
    }
  }

  // Render Live Result UI
  function renderLiveResult(res) {
    document.getElementById('result-placeholder').classList.add('hidden');
    const content = document.getElementById('result-content');
    content.classList.remove('hidden');

    const isCb = res.is_clickbait;
    const prob = res.clickbait_probability;
    const legitProb = res.legitimate_probability;

    // Header Risk Badge
    const riskBadge = document.getElementById('risk-badge');
    riskBadge.textContent = res.risk_level;
    riskBadge.className = `badge ${isCb ? 'badge-danger' : 'badge-success'}`;

    // Verdict Banner
    const verdictTag = document.getElementById('verdict-tag');
    verdictTag.textContent = res.prediction.toUpperCase();
    verdictTag.style.color = res.risk_color;

    document.getElementById('verdict-headline').textContent = res.headline;
    document.getElementById('verdict-desc').textContent = res.verdict;

    // Circular Gauge Animation
    const gaugeCircle = document.getElementById('gauge-circle');
    const gaugePct = document.getElementById('gauge-pct');
    
    gaugeCircle.style.stroke = res.risk_color;
    const offset = CIRCLE_CIRCUMFERENCE - (prob / 100) * CIRCLE_CIRCUMFERENCE;
    gaugeCircle.style.strokeDashoffset = offset;
    gaugePct.textContent = prob;

    // Split Bar
    document.getElementById('val-cb-prob').textContent = `${prob}%`;
    document.getElementById('val-legit-prob').textContent = `${legitProb}%`;
    document.getElementById('bar-fill-cb').style.width = `${prob}%`;
    document.getElementById('bar-fill-legit').style.width = `${legitProb}%`;

    // Tokens Stream Highlighting
    const tokenStream = document.getElementById('token-stream');
    tokenStream.innerHTML = '';
    (res.highlighted_tokens || []).forEach(t => {
      const span = document.createElement('span');
      span.className = `token ${t.type}`;
      span.textContent = t.word;
      tokenStream.appendChild(span);
    });

    // Linguistic Grid
    const ling = res.linguistic_breakdown || {};
    document.getElementById('metric-sensationalism').textContent = `${ling.sensationalism_score || 0} / 10`;
    document.getElementById('metric-caps').textContent = `${ling.capitalization_pct || 0}% (${ling.all_caps_words || 0} caps words)`;
    document.getElementById('metric-punct').textContent = `! : ${ling.exclamation_marks || 0} | ? : ${ling.question_marks || 0}`;
    document.getElementById('metric-listicle').textContent = ling.listicle_pattern ? 'Listicle Detected (High Risk)' : 'Standard Format';
    document.getElementById('metric-triggers').textContent = `${ling.clickbait_triggers_found || 0} trigger patterns`;
    document.getElementById('metric-counts').textContent = `${ling.word_count || 0} words (${ling.char_length || 0} chars)`;

    // AI Rephrase Box
    const rewriteContainer = document.getElementById('rewrite-container');
    const rewriteText = document.getElementById('rewrite-text');
    if (isCb && res.neutral_rewrite) {
      rewriteContainer.classList.remove('hidden');
      rewriteText.textContent = `"${res.neutral_rewrite}"`;
    } else {
      rewriteContainer.classList.add('hidden');
    }
  }

  // Batch Processor Logic
  function initBatchProcessor() {
    const dropZone = document.getElementById('csv-drop-zone');
    const fileInput = document.getElementById('csv-file-input');
    const fileNameLabel = document.getElementById('file-name-label');
    const batchBtn = document.getElementById('batch-run-btn');
    const sampleBatchBtn = document.getElementById('load-sample-batch-btn');
    const exportBtn = document.getElementById('export-csv-btn');
    const tableSearch = document.getElementById('table-search');
    const filterBtns = document.querySelectorAll('.filter-btn');

    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileNameLabel.textContent = fileInput.files[0].name;
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) {
        fileNameLabel.textContent = fileInput.files[0].name;
      }
    });

    sampleBatchBtn.addEventListener('click', () => {
      const sampleLines = [
        "15 Mind-Blowing Facts You Won't Believe Actually Happened!",
        "Federal Reserve Holds Interest Rates Steady At 5.25 Percent",
        "She Opened The Mystery Box And What Happened Next Left Everyone In Tears",
        "NASA Launches Artemis Spacecraft On Lunar Exploration Mission",
        "Top 10 Secrets Doctors Don't Want You To Know About Weight Loss",
        "United Nations Summit Concludes With Global Climate Accord",
        "Why Millenials Are Completely Ruining The Diamond Industry",
        "Tech Giant Unveils Quantum Computing Chip With 1000 Qubits",
        "Stop Doing This One Dangerous Thing Before Going To Sleep",
        "Astronomers Discover Water Vapor Signatures In Exoplanet Atmosphere"
      ];
      document.getElementById('batch-textarea').value = sampleLines.join('\n');
    });

    batchBtn.addEventListener('click', async () => {
      let bodyData = null;
      let headers = {};

      if (fileInput.files.length > 0) {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        bodyData = formData;
      } else {
        const text = document.getElementById('batch-textarea').value.trim();
        if (!text) {
          alert('Please either upload a CSV file or paste headlines into the textarea.');
          return;
        }
        const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        bodyData = JSON.stringify({ headlines: lines });
        headers = { 'Content-Type': 'application/json' };
      }

      batchBtn.disabled = true;
      batchBtn.textContent = 'Processing Batch...';

      try {
        const resp = await fetch('/api/batch', {
          method: 'POST',
          headers,
          body: bodyData
        });
        const res = await resp.json();
        if (res.error) {
          alert(res.error);
          return;
        }
        batchData = res.results || [];
        renderBatchResults(res);
      } catch (err) {
        console.error('Batch error:', err);
        alert('Batch processing failed.');
      } finally {
        batchBtn.disabled = false;
        batchBtn.textContent = 'Process Batch Headlines';
      }
    });

    // Table Filter and Search
    tableSearch.addEventListener('input', () => filterAndRenderBatchTable());
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        filterAndRenderBatchTable();
      });
    });

    // CSV Exporter
    exportBtn.addEventListener('click', () => {
      if (batchData.length === 0) return;
      let csvContent = "data:text/csv;charset=utf-8,Headline,Classification,Clickbait_Probability,Risk_Level\n";
      batchData.forEach(row => {
        const cleanHeadline = `"${row.headline.replace(/"/g, '""')}"`;
        csvContent += `${cleanHeadline},${row.prediction},${row.clickbait_probability}%,${row.risk_level}\n`;
      });
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "clickbait_analysis_results.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  function renderBatchResults(data) {
    document.getElementById('batch-results-card').classList.remove('hidden');
    const sum = data.summary;

    document.getElementById('stat-total').textContent = sum.total_evaluated;
    document.getElementById('stat-cb-count').textContent = sum.clickbait_count;
    document.getElementById('stat-cb-pct').textContent = `${sum.clickbait_percentage}%`;
    document.getElementById('stat-legit-count').textContent = sum.legitimate_count;
    document.getElementById('stat-legit-pct').textContent = `${sum.legitimate_percentage}%`;

    filterAndRenderBatchTable();
  }

  function filterAndRenderBatchTable() {
    const searchVal = document.getElementById('table-search').value.toLowerCase();
    const activeFilter = document.querySelector('.filter-btn.active').getAttribute('data-filter');
    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';

    const filtered = batchData.filter(item => {
      const matchesSearch = item.headline.toLowerCase().includes(searchVal);
      if (!matchesSearch) return false;
      if (activeFilter === 'clickbait') return item.is_clickbait;
      if (activeFilter === 'legitimate') return !item.is_clickbait;
      return true;
    });

    filtered.forEach((item, idx) => {
      const tr = document.createElement('tr');
      const badgeClass = item.is_clickbait ? 'badge-danger' : 'badge-success';
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td><strong>${escapeHtml(item.headline)}</strong></td>
        <td><span class="badge ${badgeClass}">${item.prediction}</span></td>
        <td><strong>${item.clickbait_probability}%</strong></td>
        <td style="color: ${item.risk_color}; font-weight: 600;">${item.risk_level}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Model Benchmarks Loader
  async function loadModelBenchmarks() {
    try {
      const resp = await fetch('/api/metrics');
      const data = await resp.json();
      globalMetrics = data;
      renderModelBenchmarks(data);
    } catch (e) {
      console.error('Failed to load metrics:', e);
    }
  }

  function renderModelBenchmarks(data) {
    if (!data.models) return;
    const cardsRow = document.getElementById('models-cards-row');
    cardsRow.innerHTML = '';

    const select = document.getElementById('matrix-model-select');
    select.innerHTML = '';

    const bestModel = data.best_model;
    document.getElementById('active-model-name').textContent = bestModel;

    Object.entries(data.models).forEach(([name, m]) => {
      const isBest = name === bestModel;
      const card = document.createElement('div');
      card.className = `model-card ${isBest ? 'featured' : ''}`;
      card.innerHTML = `
        <div>
          <h3 class="model-name">${name}</h3>
          <div class="model-metrics-list">
            <div class="metric-row"><span class="metric-name">Accuracy</span><span class="metric-val" style="color: #34d399;">${m.accuracy}%</span></div>
            <div class="metric-row"><span class="metric-name">Precision</span><span class="metric-val">${m.precision}%</span></div>
            <div class="metric-row"><span class="metric-name">Recall</span><span class="metric-val">${m.recall}%</span></div>
            <div class="metric-row"><span class="metric-name">F1-Score</span><span class="metric-val" style="color: #a78bfa;">${m.f1_score}%</span></div>
            <div class="metric-row"><span class="metric-name">ROC-AUC</span><span class="metric-val">${m.roc_auc}%</span></div>
          </div>
        </div>
      `;
      cardsRow.appendChild(card);

      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (isBest) opt.selected = true;
      select.appendChild(opt);
    });

    select.addEventListener('change', () => {
      updateConfusionMatrix(select.value);
    });
    updateConfusionMatrix(bestModel);

    // Render Top Feature Keywords
    const kwList = document.getElementById('keywords-list');
    kwList.innerHTML = '';
    (data.top_clickbait_keywords || []).slice(0, 15).forEach(kw => {
      const div = document.createElement('div');
      div.className = 'keyword-item';
      div.innerHTML = `
        <span class="kw-term">"${kw.word}"</span>
        <span class="kw-weight">+${kw.weight}</span>
      `;
      kwList.appendChild(div);
    });
  }

  function updateConfusionMatrix(modelName) {
    if (!globalMetrics || !globalMetrics.models[modelName]) return;
    const cm = globalMetrics.models[modelName].confusion_matrix;
    if (cm && cm.length === 2) {
      document.getElementById('cm-tn').querySelector('.cm-val').textContent = cm[0][0];
      document.getElementById('cm-fp').querySelector('.cm-val').textContent = cm[0][1];
      document.getElementById('cm-fn').querySelector('.cm-val').textContent = cm[1][0];
      document.getElementById('cm-tp').querySelector('.cm-val').textContent = cm[1][1];
    }
  }

  // Dataset Explorer Loader
  async function loadDatasetExplorer() {
    try {
      const resp = await fetch('/api/dataset-stats');
      const data = await resp.json();
      if (data.total_records) {
        document.getElementById('ds-total').textContent = data.total_records.toLocaleString();
        document.getElementById('ds-cb').textContent = data.clickbait_count.toLocaleString();
        document.getElementById('ds-legit').textContent = data.legitimate_count.toLocaleString();
        datasetRows = data.sample_rows || [];
        renderDatasetTable(datasetRows);
      }
    } catch (e) {
      console.error('Failed to load dataset stats:', e);
    }
  }

  function renderDatasetTable(rows) {
    const tbody = document.getElementById('dataset-table-body');
    const search = document.getElementById('dataset-search');
    
    function draw(filterText = '') {
      tbody.innerHTML = '';
      rows.filter(r => (r.headline || '').toLowerCase().includes(filterText.toLowerCase())).forEach(r => {
        const tr = document.createElement('tr');
        const isCb = r.label === 1;
        tr.innerHTML = `
          <td><strong>${escapeHtml(r.headline)}</strong></td>
          <td>${escapeHtml(r.category || 'News')}</td>
          <td><span class="badge ${isCb ? 'badge-danger' : 'badge-success'}">${isCb ? 'Clickbait' : 'Legitimate'}</span></td>
        `;
        tbody.appendChild(tr);
      });
    }

    search.addEventListener('input', (e) => draw(e.target.value));
    draw();
  }

  // Initial Data Fetch
  async function fetchInitialData() {
    try {
      const resp = await fetch('/api/sample-headlines');
      sampleHeadlinesList = await resp.json();
      loadModelBenchmarks();
    } catch (e) {
      console.warn('Initial data preload:', e);
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
});
