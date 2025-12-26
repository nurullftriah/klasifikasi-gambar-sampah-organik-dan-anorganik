const elFile = document.getElementById('file');
const elPreview = document.getElementById('preview');
const elPlaceholder = document.getElementById('placeholder');
const elBtnPredict = document.getElementById('btnPredict');
const elBtnReset = document.getElementById('btnReset');
const elStatus = document.getElementById('status');
const elResult = document.getElementById('result');

let selectedFile = null;

function setStatus(msg, type = '') {
    elStatus.textContent = msg || '';
    elStatus.className = 'status ' + (type ? type : '');
}

function showResult(d) {
    elResult.classList.remove('hidden');
    const pct = x => (x * 100).toFixed(2) + '%';
    elResult.innerHTML = `
    <div class="badge ${d.label === 'organik' ? 'green' : 'blue'}">
      ${d.label.toUpperCase()}
    </div>
    <div class="kv">
      <div><span>Prob. Organik</span><b>${pct(d.prob_organik)}</b></div>
      <div><span>Confidence</span><b>${pct(d.confidence)}</b></div>
    </div>
  `;
}

function resetUI() {
    selectedFile = null;
    elFile.value = '';
    elPreview.src = '';
    elPreview.classList.add('hidden');
    elPlaceholder.classList.remove('hidden');
    elBtnPredict.disabled = true;
    elBtnReset.disabled = true;
    elResult.classList.add('hidden');
    elResult.innerHTML = '';
    setStatus('');
}

elFile.addEventListener('change', () => {
    const f = elFile.files && elFile.files[0];
    if (!f) return resetUI();

    if (f.size > 5 * 1024 * 1024) {
        resetUI();
        return setStatus('File terlalu besar. Maks 5MB.', 'err');
    }

    selectedFile = f;
    elBtnPredict.disabled = false;
    elBtnReset.disabled = false;

    const url = URL.createObjectURL(f);
    elPreview.src = url;
    elPreview.classList.remove('hidden');
    elPlaceholder.classList.add('hidden');
    setStatus('Siap diprediksi.');
});

elBtnReset.addEventListener('click', resetUI);

elBtnPredict.addEventListener('click', async () => {
    if (!selectedFile) return;

    setStatus('Memproses prediksi...', 'loading');
    elBtnPredict.disabled = true;

    try {
        const fd = new FormData();
        fd.append('file', selectedFile);

        const res = await fetch('/api/predict', { method: 'POST', body: fd });
        const data = await res.json();

        if (!res.ok) throw new Error(data.error || 'Gagal prediksi');

        showResult(data);
        setStatus('Selesai.');
    } catch (e) {
        setStatus(String(e.message || e), 'err');
    } finally {
        elBtnPredict.disabled = false;
    }
});

resetUI();
