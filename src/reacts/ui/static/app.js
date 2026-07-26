async function getJSON(url, options={}) {
  const response = await fetch(url, {headers:{"Content-Type":"application/json"}, ...options});
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}
function metrics(obj) {
  if (!obj) return '<p class="muted">Not built yet.</p>';
  const m = obj.metrics || {};
  return Object.entries(m).slice(0,12).map(([k,v])=>`<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');
}
async function load() {
  try { const h=await getJSON('/health'); document.querySelector('#health').textContent=`${h.status} · ${h.models} models`; } catch(e){document.querySelector('#health').textContent=e.message}
  try { const d=await getJSON('/api/v1/datasets'); document.querySelector('#dataset').innerHTML=metrics(d.active); } catch(e){document.querySelector('#dataset').textContent=e.message}
  try { const ms=await getJSON('/api/v1/models'); document.querySelector('#models').innerHTML=ms.length?ms.map(m=>`<div class="metric"><span>${m.task}</span><strong>${m.stage}</strong></div>`).join(''):'<p class="muted">No models registered.</p>'; } catch(e){document.querySelector('#models').textContent=e.message}
}
document.querySelector('#predict').addEventListener('click', async ()=>{
  const tasks=[...document.querySelectorAll('.controls input:checked')].map(x=>x.value);
  const reaction_smiles=document.querySelector('#reaction').value.trim();
  const out=document.querySelector('#prediction'); out.textContent='Running…';
  try { out.textContent=JSON.stringify(await getJSON('/api/v1/inference/conditions',{method:'POST',body:JSON.stringify({reaction_smiles,tasks,include_evidence:true,evidence_k:5})}),null,2); } catch(e){out.textContent=e.message}
});
load();
