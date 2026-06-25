/* ═══════════════════════════════════════════════════════════════════════════════
   Chennai HRES · AI Energy Platform — Chart.js Charts + Animated Counters
   ═══════════════════════════════════════════════════════════════════════════════ */

// ── COLOUR PALETTE ──────────────────────────────────────────────────────────────
const C = {
  cyan:   '#00D9FF', yellow: '#FFD54A', green:  '#4ADE80',
  orange: '#FB923C', blue:   '#60A5FA', red:    '#EF4444',
  text:   '#E2E8F0', muted:  '#64748B', grid:   '#1E293B',
  paper:  '#0F1629',
};

// ── PRECOMPUTED SIMULATION DATA ─────────────────────────────────────────────────
// These values come from the simulation run (November 7, 2026 — NE Monsoon Day)
const HOURS = ['00','01','02','03','04','05','06','07','08','09','10','11',
               '12','13','14','15','16','17','18','19','20','21','22','23'];

// Solar PV output (kW) — NE Monsoon day: very low due to thick cloud cover
const SOLAR_KW = [0,0,0,0,0,0,0.2,1.8,5.3,8.7,11.2,12.4,
                  10.8,9.1,6.5,3.9,1.2,0.1,0,0,0,0,0,0];

// TENG output (kW) — heavy cyclonic rain throughout the day
const TENG_KW = [2.1,1.8,3.5,5.2,4.1,6.8,8.3,3.2,1.5,0.8,2.4,3.8,
                 5.1,7.9,9.2,11.5,8.7,6.4,4.2,7.1,9.8,5.6,3.1,2.3];

// School load (kW) — academic day
const LOAD_KW = [7.2,6.8,6.5,6.9,7.1,7.4,8.2,22.5,62.3,68.1,71.4,72.8,
                 65.2,69.7,67.3,58.4,24.1,10.2,8.5,7.8,7.5,7.2,7.0,6.8];

// Battery SOC (%) — starts at 50%, charges/discharges throughout day
const BAT_SOC = [50,48,46,44,42,40,39,35,28,22,20,21,
                 23,22,20,20,22,28,34,38,42,45,47,49];

// LSTM predicted load (kW) — close to actual with ~3 kW MAE noise
const LSTM_PRED = [8.1,7.5,6.2,7.4,6.8,8.0,9.1,24.3,59.8,70.2,69.1,74.5,
                   63.8,67.2,69.8,56.1,22.5,11.8,7.9,8.5,6.9,8.1,7.5,7.2];

// Monthly generation data (kWh)
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const MONTHLY_SOLAR = [14200,13800,16500,15800,14200,5800,4200,5100,6200,3800,2200,3100];
const MONTHLY_TENG  = [15,10,0,0,0,180,320,280,210,420,680,520];
const MONTHLY_GRID  = [2800,2200,1500,2100,8200,9800,11200,10500,9200,12800,14500,12200];
const MONTHLY_UNMET = [0,0,0,0,0,200,400,350,280,3200,8500,5800];
const MONTHLY_LOAD  = [15800,14500,16200,16800,20500,16200,16500,16800,16200,18500,22800,19500];

// Net energy (surplus/deficit) per hour
const NET_ENERGY = SOLAR_KW.map((s, i) => s + TENG_KW[i] - LOAD_KW[i]);

// Rainfall vs TENG theoretical curve
const RAIN_RATES = [];
const TENG_THEORETICAL = [];
for (let r = 0; r <= 80; r += 2) {
  RAIN_RATES.push(r);
  // Simplified TENG power model: P ≈ k * R^1.3 (fitted to simulation results)
  const p = r === 0 ? 0 : 0.018 * Math.pow(r, 1.3);
  TENG_THEORETICAL.push(parseFloat(p.toFixed(2)));
}

// ── CHART.JS GLOBAL CONFIG ──────────────────────────────────────────────────────
Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.grid;
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 22, 41, 0.95)';
Chart.defaults.plugins.tooltip.borderColor = C.grid;
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.titleFont = { family: "'Inter'", weight: '600', size: 12 };
Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono'", size: 11 };
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.animation.duration = 1500;
Chart.defaults.animation.easing = 'easeOutQuart';

// ── CHART 1: SOLAR vs TENG 24h ─────────────────────────────────────────────────
new Chart(document.getElementById('chart-solar-teng'), {
  type: 'line',
  data: {
    labels: HOURS.map(h => h + ':00'),
    datasets: [
      {
        label: 'Solar PV (kW)',
        data: SOLAR_KW,
        borderColor: C.yellow,
        backgroundColor: 'rgba(255, 213, 74, 0.08)',
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: C.yellow,
      },
      {
        label: 'TENG Rain (kW)',
        data: TENG_KW,
        borderColor: C.cyan,
        backgroundColor: 'rgba(0, 217, 255, 0.08)',
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: C.cyan,
      },
      {
        label: 'School Load (kW)',
        data: LOAD_KW,
        borderColor: C.orange,
        borderWidth: 2,
        borderDash: [6, 3],
        fill: false,
        tension: 0.4,
        pointRadius: 2,
        pointBackgroundColor: C.orange,
      }
    ]
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: { grid: { color: C.grid } },
      y: { grid: { color: C.grid }, title: { display: true, text: 'Power (kW)', color: C.muted } }
    }
  }
});

// ── CHART 2: BATTERY SOC ────────────────────────────────────────────────────────
new Chart(document.getElementById('chart-battery-soc'), {
  type: 'line',
  data: {
    labels: HOURS.map(h => h + ':00'),
    datasets: [{
      label: 'Battery SOC (%)',
      data: BAT_SOC,
      borderColor: C.green,
      backgroundColor: 'rgba(74, 222, 128, 0.1)',
      borderWidth: 3,
      fill: true,
      tension: 0.4,
      pointRadius: 3,
      pointBackgroundColor: C.green,
    }]
  },
  options: {
    responsive: true,
    scales: {
      x: { grid: { color: C.grid } },
      y: {
        min: 0, max: 100,
        grid: { color: C.grid },
        title: { display: true, text: 'SOC (%)', color: C.muted }
      }
    },
    plugins: {
      annotation: {
        annotations: {
          minSOC: {
            type: 'line', yMin: 20, yMax: 20,
            borderColor: C.red, borderWidth: 1, borderDash: [5, 5],
            label: { display: true, content: 'Min SOC (20%)', color: C.red, font: { size: 10 } }
          }
        }
      }
    }
  }
});

// ── CHART 3: MONTHLY GENERATION ─────────────────────────────────────────────────
new Chart(document.getElementById('chart-monthly'), {
  type: 'bar',
  data: {
    labels: MONTHS,
    datasets: [
      { label: 'Solar PV',    data: MONTHLY_SOLAR, backgroundColor: C.yellow + 'CC', borderRadius: 4, barPercentage: 0.7 },
      { label: 'TENG',        data: MONTHLY_TENG,  backgroundColor: C.cyan + 'CC',   borderRadius: 4, barPercentage: 0.7 },
      { label: 'Grid Import', data: MONTHLY_GRID,  backgroundColor: C.blue + '99',   borderRadius: 4, barPercentage: 0.7 },
      { label: 'Unmet Load',  data: MONTHLY_UNMET, backgroundColor: C.red + '99',    borderRadius: 4, barPercentage: 0.7 },
    ]
  },
  options: {
    responsive: true,
    scales: {
      x: { stacked: true, grid: { color: C.grid } },
      y: { stacked: true, grid: { color: C.grid }, title: { display: true, text: 'Energy (kWh)', color: C.muted } }
    }
  }
});

// ── CHART 4: SURPLUS & DEFICIT ──────────────────────────────────────────────────
const surplusData = NET_ENERGY.map(v => v >= 0 ? v : 0);
const deficitData = NET_ENERGY.map(v => v < 0 ? v : 0);

new Chart(document.getElementById('chart-surplus-deficit'), {
  type: 'bar',
  data: {
    labels: HOURS.map(h => h + ':00'),
    datasets: [
      { label: 'Surplus (kW)',  data: surplusData, backgroundColor: C.green + 'BB', borderRadius: 3 },
      { label: 'Deficit (kW)',  data: deficitData, backgroundColor: C.red + 'BB',   borderRadius: 3 },
    ]
  },
  options: {
    responsive: true,
    scales: {
      x: { grid: { color: C.grid } },
      y: { grid: { color: C.grid }, title: { display: true, text: 'Net Power (kW)', color: C.muted } }
    }
  }
});

// ── CHART 5: RAINFALL vs TENG CURVE ─────────────────────────────────────────────
new Chart(document.getElementById('chart-rainfall-teng'), {
  type: 'line',
  data: {
    labels: RAIN_RATES.map(r => r + ' mm/h'),
    datasets: [{
      label: 'TENG Power Output (kW)',
      data: TENG_THEORETICAL,
      borderColor: C.cyan,
      backgroundColor: 'rgba(0, 217, 255, 0.08)',
      borderWidth: 3,
      fill: true,
      tension: 0.4,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: C.cyan,
    }]
  },
  options: {
    responsive: true,
    scales: {
      x: { grid: { color: C.grid }, title: { display: true, text: 'Rainfall Intensity (mm/h)', color: C.muted },
           ticks: { maxTicksLimit: 10 } },
      y: { grid: { color: C.grid }, title: { display: true, text: 'Power (kW)', color: C.muted } }
    }
  }
});

// ── CHART 6: AI FORECAST ────────────────────────────────────────────────────────
new Chart(document.getElementById('chart-ai-forecast'), {
  type: 'line',
  data: {
    labels: HOURS.map(h => h + ':00'),
    datasets: [
      {
        label: 'Actual Load (kW)',
        data: LOAD_KW,
        borderColor: C.orange,
        borderWidth: 2.5,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: C.orange,
      },
      {
        label: 'LSTM Predicted (kW)',
        data: LSTM_PRED,
        borderColor: C.cyan,
        borderWidth: 2,
        borderDash: [6, 3],
        tension: 0.4,
        pointRadius: 2,
        pointBackgroundColor: C.cyan,
      },
      {
        label: '+1σ Confidence',
        data: LSTM_PRED.map(v => v + 5.51),
        borderColor: 'transparent',
        backgroundColor: 'rgba(0, 217, 255, 0.06)',
        borderWidth: 0,
        fill: '+1',
        tension: 0.4,
        pointRadius: 0,
      },
      {
        label: '-1σ Confidence',
        data: LSTM_PRED.map(v => Math.max(0, v - 5.51)),
        borderColor: 'transparent',
        backgroundColor: 'rgba(0, 217, 255, 0.06)',
        borderWidth: 0,
        fill: '-1',
        tension: 0.4,
        pointRadius: 0,
      },
    ]
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: { grid: { color: C.grid } },
      y: { grid: { color: C.grid }, title: { display: true, text: 'Load (kW)', color: C.muted } }
    }
  }
});

// ── ANIMATED COUNTERS ───────────────────────────────────────────────────────────
function animateCounters() {
  const counters = document.querySelectorAll('.counter');
  counters.forEach(counter => {
    const target = parseFloat(counter.getAttribute('data-target'));
    const decimals = parseInt(counter.getAttribute('data-decimals') || '0');
    const duration = 2000;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutQuart
      const eased = 1 - Math.pow(1 - progress, 4);
      const current = eased * target;

      if (decimals > 0) {
        counter.textContent = current.toFixed(decimals);
      } else {
        counter.textContent = Math.floor(current).toLocaleString('en-IN');
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        if (decimals > 0) {
          counter.textContent = target.toFixed(decimals);
        } else {
          counter.textContent = Math.floor(target).toLocaleString('en-IN');
        }
      }
    }
    requestAnimationFrame(update);
  });
}

// ── LIVE CLOCK ──────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const opts = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' };
  const dateOpts = { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'Asia/Kolkata' };
  const timeStr = now.toLocaleTimeString('en-IN', opts);
  const dateStr = now.toLocaleDateString('en-IN', dateOpts);
  document.getElementById('live-clock').textContent = `${dateStr} · ${timeStr} IST`;
}

// ── INTERSECTION OBSERVER (trigger counters on scroll) ──────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounters();
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.3 });

document.addEventListener('DOMContentLoaded', () => {
  // Start clock
  updateClock();
  setInterval(updateClock, 1000);

  // Observe KPI section for counter animation
  const kpiSection = document.getElementById('kpi-section');
  if (kpiSection) {
    observer.observe(kpiSection);
  }
});
