// Colors mapped from backend payload CATEGORY_COLORS
const CAT = {
    Structure: 0x6b7280, Memory: 0x3b82f6, Cognition: 0xeab308, Consciousness: 0xa855f7,
    Learning: 0x60a5fa, Perception: 0x22c55e, Defense: 0xef4444, Evolution: 0xf97316,
    Synthesis: 0x06b6d4, Resilience: 0x14b8a6, Social: 0xec4899, Actuation: 0xfbbf24,
    Autos: 0xf8fafc, Unknown: 0x9ca3af
};

// --- Three.js Setup ---
const container = document.getElementById("tree-container");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 3000);
// Top-down tiled isometric-ish angle
camera.position.set(0, 1000, 300);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
// Disable spin, lock the motherboard angle
controls.autoRotate = false;
controls.maxPolarAngle = Math.PI / 2.5;

const ambientLight = new THREE.AmbientLight(0x222222);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
dirLight.position.set(200, 500, 300);
scene.add(dirLight);

// --- Object Management ---
let nodesMap = {}; 
let interactables = [];
let particles = [];
let drawnLinks = [];

const sphereGeo = new THREE.SphereGeometry(22, 32, 32);
const lineMat = new THREE.LineBasicMaterial({ color: 0x6b7280, transparent: true, opacity: 0.3 });

window.rebuildTopology = function(data) {
    if (!data || !data.nodes) return;
    
    // Cleanup existing geometry
    for (const key in nodesMap) {
        scene.remove(nodesMap[key]);
    }
    drawnLinks.forEach(l => scene.remove(l.mesh));
    interactables = [];
    nodesMap = {};
    drawnLinks = [];

    // Render LLM Dynamic Nodes
    data.nodes.forEach(n => {
        const group = new THREE.Group();
        group.position.set(n.pos[0] || 0, n.pos[1] || 0, n.pos[2] || 0);

        // Motherboard Chips (Squares)
        let geo = new THREE.BoxGeometry(45, 6, 45);

        const hex = CAT[n.category] || CAT.Unknown;
        const mat = new THREE.MeshStandardMaterial({ color: hex, roughness: 0.2, metalness: 0.9, emissive: hex, emissiveIntensity: 0.2 });
        const wireMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.1 });
        
        const coreMesh = new THREE.Mesh(geo, mat);
        const wire = new THREE.Mesh(geo, wireMat);
        wire.scale.set(1.1, 1.1, 1.1);

        const light = new THREE.PointLight(hex, 0.4, 150);
        light.position.set(0, 20, 0);
        
        coreMesh.userData = { type: "core", id: n.id, category: n.category };
        interactables.push(coreMesh);

        // Name tag sprite
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.font = '24px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText(n.id.toUpperCase(), 128, 40);
        
        const tex = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, opacity:0.8 });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(100, 25, 1);
        sprite.position.y = 20;
        sprite.position.z = 35; // Put label below the chip

        group.add(coreMesh);
        group.add(wire);
        group.add(light);
        group.add(sprite);
        
        scene.add(group);
        nodesMap[n.id] = group;
        
        // Procedural Embedded Grid Sub-Nodes (Functions)
        if (n.functions && n.functions.length > 0) {
            const funcGeo = new THREE.BoxGeometry(6, 2, 6);
            const funcMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: false, transparent: true, opacity: 0.3 });
            
            // Grid layout calculation
            const cols = Math.ceil(Math.sqrt(n.functions.length));
            const spacing = 12;
            const startX = -((cols - 1) * spacing) / 2;
            const startZ = -((Math.ceil(n.functions.length / cols) - 1) * spacing) / 2;
            
            n.functions.forEach((fName, i) => {
                const col = i % cols;
                const row = Math.floor(i / cols);
                
                const fx = startX + (col * spacing);
                const fz = startZ + (row * spacing);
                const fy = 4; // Slightly embedded top of chip surface
                
                const fMesh = new THREE.Mesh(funcGeo, funcMat.clone());
                fMesh.position.set(fx, fy, fz);
                
                fMesh.userData = { type: "function", parent: n.id, id: fName, category: n.category || "Unknown" };
                interactables.push(fMesh);
                
                group.add(fMesh);
                nodesMap[`${n.id}:${fName}`] = fMesh;
                
                // Truncate function names for the tiny chip labels
                const tinyName = fName.length > 5 ? fName.substring(0,3) + '..' : fName;
                const fCanvas = document.createElement('canvas');
                fCanvas.width = 128; fCanvas.height = 32;
                const fCtx = fCanvas.getContext('2d');
                fCtx.fillStyle = '#09090b'; // dark text on lit chip
                fCtx.font = '16px "JetBrains Mono"';
                fCtx.textAlign = 'center';
                fCtx.fillText(tinyName, 64, 20);
                
                const fTex = new THREE.CanvasTexture(fCanvas);
                const fSpriteMat = new THREE.SpriteMaterial({ map: fTex, transparent: true, opacity:0.8, depthTest: false });
                const fSprite = new THREE.Sprite(fSpriteMat);
                fSprite.scale.set(16, 4, 1);
                fSprite.position.set(fx, fy + 2, fz);
                group.add(fSprite);
            });
        }
    });

    // Render LLM Dynamic Links - with procedural failsafe
    if (!data.links || data.links.length === 0) {
        data.links = [];
        const ids = data.nodes.map(n => n.id);
        for (let i = 0; i < ids.length; i++) {
            if (i > 0) data.links.push([ids[i-1], ids[i]]);
            if (i > 2) data.links.push([ids[i-3], ids[i]]);
        }
    }

    if(data.links) {
        data.links.forEach(l => {
            const s = nodesMap[l[0]]?.position;
            const t = nodesMap[l[1]]?.position;
            if(!s || !t) return;
            
            // Manhattan Routing (L-Shaped Traces) for the Motherboard
            const points = [];
            points.push(new THREE.Vector3(s.x, s.y - 2, s.z));
            if (Math.abs(s.x - t.x) > Math.abs(s.z - t.z)) {
                points.push(new THREE.Vector3((s.x + t.x) / 2, s.y - 2, s.z));
                points.push(new THREE.Vector3((s.x + t.x) / 2, t.y - 2, t.z));
            } else {
                points.push(new THREE.Vector3(s.x, s.y - 2, (s.z + t.z) / 2));
                points.push(new THREE.Vector3(t.x, t.y - 2, (s.z + t.z) / 2));
            }
            points.push(new THREE.Vector3(t.x, t.y - 2, t.z));

            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geo, lineMat.clone());
            scene.add(line);
            
            // Generate full path cache for particle signals so they can follow the L-shape correctly
            drawnLinks.push({ mesh: line, source: l[0], target: l[1], path: points });
        });
    }
    console.log("Topology rebuilt fully from backend LLM instructions.", Object.keys(nodesMap).length, "nodes rendering.");
};

// --- Mouse Raycaster ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

const tooltip = document.createElement("div");
tooltip.style.position = "absolute";
tooltip.style.padding = "8px 12px";
tooltip.style.background = "rgba(15, 23, 42, 0.9)";
tooltip.style.border = "1px solid rgba(255,255,255,0.2)";
tooltip.style.color = "#f8fafc";
tooltip.style.fontFamily = "monospace";
tooltip.style.borderRadius = "4px";
tooltip.style.pointerEvents = "none";
tooltip.style.opacity = "0";
tooltip.style.transition = "opacity 0.2s";
tooltip.style.zIndex = "1000";
document.body.appendChild(tooltip);

window.addEventListener("mousemove", (e) => {
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(interactables);
    
    document.body.style.cursor = "default";
    tooltip.style.opacity = "0";

    if (intersects.length > 0) {
        document.body.style.cursor = "pointer";
        const tar = intersects[0].object;
        
        tooltip.innerHTML = `Organ: <strong>${tar.userData.id.toUpperCase()}</strong><br>Type: ${tar.userData.category}<br><em>Click to inject</em>`;
        tooltip.style.left = (e.clientX + 15) + "px";
        tooltip.style.top = (e.clientY + 15) + "px";
        tooltip.style.opacity = "1";
    }
});

window.resetIsolationMode = function() {
    drawnLinks.forEach(l => {
        l.mesh.material.opacity = 0.3;
        l.mesh.material.color.setHex(0x6b7280);
    });
};

window.addEventListener("click", (e) => {
    if (e.target.tagName !== "CANVAS") return;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(interactables);
    if (intersects.length > 0) {
        const tar = intersects[0].object;
        handleNodeClick(tar);
    } else {
        window.resetIsolationMode();
    }
});

// --- Dynamic Signaling Engine ---
window.spawnSignalParticle = function(sourceId, targetId, colorHex=0xffffff) {
    if(!nodesMap[sourceId] || !nodesMap[targetId]) return;
    
    // Find the routing path to follow
    const link = drawnLinks.find(l => (l.source === sourceId && l.target === targetId) || (l.target === sourceId && l.source === targetId));
    if (!link) return;

    let path = [...link.path];
    if (link.source !== sourceId) path.reverse(); // If going backwards along the trace
    
    const pGeo = new THREE.SphereGeometry(6, 12, 12);
    const pMat = new THREE.MeshBasicMaterial({ color: colorHex });
    const pMesh = new THREE.Mesh(pGeo, pMat);
    pMesh.position.copy(path[0]);
    scene.add(pMesh);
    
    // Physics pulse the source
    pulseNode(sourceId, 1.5);
    
    particles.push({ mesh: pMesh, pathPoints: path, currentSegment: 0, progress: 0, speed: Math.random() * 0.05 + 0.03, tgt: targetId });
}

// Background idle signaling for ambient vibe
setInterval(() => {
    if(Math.random() > 0.7 && drawnLinks.length > 0) {
        const keys = Object.keys(nodesMap).filter(k => !k.includes(':'));
        if(keys.length > 1) {
            const s = keys[Math.floor(Math.random() * keys.length)];
            const t = keys[Math.floor(Math.random() * keys.length)];
            if (s !== t) window.spawnSignalParticle(s, t, 0x14b8a6); 
        }
    }
}, 1000);

window.fireFunctionSynapse = function(organId, funcName) {
    const parentGroup = nodesMap[organId];
    const funcMesh = nodesMap[`${organId}:${funcName}`];
    
    if (!parentGroup || !funcMesh) return;
    
    // Flash function sub-node
    funcMesh.material.color.setHex(0xffffff);
    funcMesh.material.opacity = 1.0;
    funcMesh.scale.set(1.5, 1.5, 1.5);
    setTimeout(() => {
        funcMesh.material.color.setHex(0xffffff);
        funcMesh.material.opacity = 0.3;
        let ticks = 0;
        const intv = setInterval(() => {
            ticks++;
            funcMesh.scale.lerp(new THREE.Vector3(1,1,1), 0.2);
            if(ticks > 15) clearInterval(intv);
        }, 16);
    }, 150);
}

// --- Animation Loop ---
function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Node Idle Breathing (Stopped for rigid grid aesthetic)

    // Process Signals along L-shaped traces
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.progress += p.speed;
        
        let segStart = p.pathPoints[p.currentSegment];
        let segEnd = p.pathPoints[p.currentSegment + 1];
        
        if (p.progress >= 1.0) {
            p.currentSegment++;
            if (p.currentSegment >= p.pathPoints.length - 1) {
                // Arrived! Pulse target
                pulseNode(p.tgt, 0.8);
                scene.remove(p.mesh);
                p.mesh.geometry.dispose();
                p.mesh.material.dispose();
                particles.splice(i, 1);
                continue;
            }
            // Move to next segment
            p.progress = 0;
            segStart = p.pathPoints[p.currentSegment];
            segEnd = p.pathPoints[p.currentSegment + 1];
        }
        
        p.mesh.position.lerpVectors(segStart, segEnd, p.progress);
    }
    renderer.render(scene, camera);
}
animate();

// --- Interactive Logic ---
function showModal(nodeObj) {
    const modal = document.getElementById("node-modal");
    if(!modal) return;
    
    const isFunc = nodeObj.userData.type === "function";
    const title = isFunc ? `${nodeObj.userData.parent}.${nodeObj.userData.id}` : nodeObj.userData.id;
    const category = nodeObj.userData.category;
    
    document.getElementById("modal-title").textContent = title.toUpperCase();
    let html = `<div class="stat-row"><span class="stat-label">Category</span><span>${category}</span></div>`;
    
    if (isFunc) {
        html += `<div class="stat-row"><span class="stat-label">Mode</span><span style="color:#22c55e">TELEMETRY TRACE</span></div>`;
        html += `<div class="stat-row"><span class="stat-label">Status</span><span>Idle / Hooked</span></div>`;
    } else if (window.lastVitals) {
        html += `<div class="stat-row"><span class="stat-label">System Uptime</span><span>${window.lastVitals.uptime_s}s</span></div>`;
        if (category === "Defense" || nodeObj.userData.id === "immune") {
            const im = window.lastVitals.immune;
            if(im) {
                html += `<div class="stat-row"><span class="stat-label">Inflammation</span><span style="color:#ef4444">${im.inflammation.toFixed(2)}</span></div>`;
                html += `<div class="stat-row"><span class="stat-label">Quarantined</span><span>${im.census?.quarantined||0}</span></div>`;
            }
        } else if (category === "Memory" || nodeObj.userData.id === "cortex") {
            const mem = window.lastVitals.memory;
            if(mem) html += `<div class="stat-row"><span class="stat-label">Fragments</span><span style="color:#3b82f6">${mem.total||0}</span></div>`;
        }
    }
    document.getElementById("modal-body").innerHTML = html;
    
    const btn = document.getElementById("modal-action-btn");
    btn.onclick = () => { 
        if (window.wsStim && window.wsStim.readyState === WebSocket.OPEN) {
            window.wsStim.send(JSON.stringify({ node: (isFunc ? nodeObj.userData.parent : nodeObj.userData.id).toLowerCase(), action: "shock" }));
            window.pulseNode(isFunc ? nodeObj.userData.parent : nodeObj.userData.id, 3.0);
        }
        modal.style.display = "none"; 
    };
    modal.style.display = "block";
}

function handleNodeClick(nodeObj) {
    const parentId = nodeObj.userData.type === "function" ? nodeObj.userData.parent : nodeObj.userData.id;
    
    // ISOLATION MODE
    let connectionTraced = false;
    drawnLinks.forEach(l => {
        if (l.source === parentId || l.target === parentId) {
            connectionTraced = true;
            l.mesh.material.opacity = 0.9;
            l.mesh.material.color.setHex(0xffffff);
            // Synaptic Wave Trigger delayed to let opacity apply visually
            setTimeout(() => {
                if (l.source === parentId) window.spawnSignalParticle(l.source, l.target, 0xffffff);
                if (l.target === parentId) window.spawnSignalParticle(l.target, l.source, 0xffffff);
            }, 50);
        } else {
            l.mesh.material.opacity = 0.05;
            l.mesh.material.color.setHex(0x111111);
        }
    });

    // If an isolated node (no links) is clicked, we still dim the rest.
    if (!connectionTraced && drawnLinks.length > 0) {
        drawnLinks.forEach(l => {
            l.mesh.material.opacity = 0.05;
            l.mesh.material.color.setHex(0x111111);
        });
    }

    showModal(nodeObj);
}

window.pulseNode = function(nodeName, intensity=1.0) {
    const group = nodesMap[nodeName];
    if(group) {
        const mesh = group.children[0];
        mesh.material.emissiveIntensity = 3 * intensity;
        mesh.scale.set(1.4, 1.4, 1.4);
        setTimeout(() => { 
            mesh.material.emissiveIntensity = 0.2; 
        }, 300);
        
        // Return scale smoothly via setInterval tick hack
        let ticks = 0;
        const intv = setInterval(() => {
            ticks++;
            mesh.scale.lerp(new THREE.Vector3(1,1,1), 0.1);
            if(ticks > 20) clearInterval(intv);
        }, 16);
    }
};

window.visualizePulse = function(data) {
    window.lastVitals = data;
    
    // Cognitive logic integration
    if (data.brain?.last_thought && nodesMap["brain"]) {
        window.pulseNode("brain", 2.0);
    }
};

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
