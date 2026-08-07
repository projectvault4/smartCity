import { useEffect, useRef, useState } from 'react';

interface CityCanvasProps {
  mode: 'home' | 'traffic' | 'air' | 'energy' | 'weather';
}

const CityCanvas = ({ mode }: CityCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tRef = useRef(0);

  const SKIES = {
    home: ['#1b4332', '#2d6a4f'],
    traffic: ['#1a1200', '#3d2b00'],
    air: ['#0a2a3a', '#1a4a6a'],
    energy: ['#1a0a3a', '#2a1560'],
    weather: ['#1a2a3a', '#243a50']
  };

  const GROUNDS = {
    home: ['#1a4a1a', '#0d2e0d'],
    traffic: ['#1a1200', '#0d0900'],
    air: ['#0a1a2a', '#050e18'],
    energy: ['#0e0820', '#070412'],
    weather: ['#111e28', '#080e14']
  };

  const rnd = (a: number, b: number) => Math.random() * (b - a) + a;

  const trees = useRef(Array.from({ length: 32 }, () => ({ x: rnd(0, 1), h: rnd(35, 70), w: rnd(18, 34), sway: rnd(0, Math.PI * 2) })));
  const clouds = useRef(Array.from({ length: 8 }, () => ({ x: rnd(0, 1.2), y: rnd(0.05, 0.2), r: rnd(28, 60), sp: rnd(0.0001, 0.00022), n: Math.ceil(rnd(3, 5)) })));
  const cars = useRef(Array.from({ length: 22 }, (_, i) => ({ lane: i % 4, x: rnd(0, 1), sp: rnd(0.0009, 0.0022), col: ['#f39c12', '#fff', '#e74c3c', '#74b9ff', '#55efc4', '#a29bfe'][i % 6] })));
  const rain = useRef(Array.from({ length: 160 }, () => ({ x: rnd(0, 1), y: rnd(0, 1), sp: rnd(0.005, 0.009), len: rnd(10, 22) })));
  const ptcls = useRef(Array.from({ length: 70 }, () => ({ x: rnd(0, 1), y: rnd(0, 1), vx: rnd(-0.001, 0.001), vy: rnd(-0.0015, -0.0004), r: rnd(1.5, 3), col: ['#55efc4', '#74b9ff', '#fdcb6e'][Math.floor(rnd(0, 3))] })));
  const birds = useRef(Array.from({ length: 6 }, () => ({ x: rnd(0, 1), y: rnd(0.05, 0.22), sp: rnd(0.0003, 0.0007), flap: rnd(0, Math.PI * 2) })));
  const bolts = useRef<{ segs: [number, number][], life: number }[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;

    const resize = () => {
      const r = canvas.parentElement?.getBoundingClientRect();
      if (r) {
        canvas.width = r.width;
        canvas.height = 520;
      }
    };

    window.addEventListener('resize', resize);
    resize();

    const drawSky = (W: number, H: number) => {
      const s = SKIES[mode];
      const g = GROUNDS[mode];
      const sky = ctx.createLinearGradient(0, 0, 0, H * 0.6);
      sky.addColorStop(0, s[0]);
      sky.addColorStop(1, s[1]);
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, W, H);
      const grd = ctx.createLinearGradient(0, H * 0.6, 0, H);
      grd.addColorStop(0, g[0]);
      grd.addColorStop(1, g[1]);
      ctx.fillStyle = grd;
      ctx.fillRect(0, H * 0.6, W, H * 0.4);
    };

    const drawClouds = (W: number, H: number) => {
      const alpha = mode === 'weather' ? 0.72 : mode === 'home' ? 0.5 : 0.25;
      clouds.current.forEach(cl => {
        cl.x = (cl.x + cl.sp) % 1.25;
        const cx = cl.x * W;
        const cy = cl.y * H;
        ctx.globalAlpha = alpha;
        for (let b = 0; b < cl.n; b++) {
          const bx = cx + b * cl.r * 0.5 - cl.r * 0.4;
          const by = cy + Math.sin(b * 1.1) * cl.r * 0.15;
          ctx.fillStyle = '#c8dcf0';
          ctx.beginPath();
          ctx.ellipse(bx, by, cl.r * (0.55 + Math.sin(b) * 0.2), cl.r * 0.38, 0, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      ctx.globalAlpha = 1;
    };

    const drawTrees = (W: number, H: number) => {
      trees.current.forEach(tr => {
        const x = tr.x * W;
        const base = H * 0.6;
        const sway = Math.sin(tRef.current * 0.5 + tr.sway) * 2;
        ctx.fillStyle = '#1a3a1a';
        ctx.fillRect(x - 2.5 + sway, base - tr.h * 0.28, 5, tr.h * 0.28);
        ctx.fillStyle = '#2d6e35';
        ctx.beginPath();
        ctx.ellipse(x + sway, base - tr.h * 0.72, tr.w / 2, tr.h * 0.56, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#3a8c42';
        ctx.beginPath();
        ctx.ellipse(x - 4 + sway, base - tr.h * 0.9, tr.w * 0.35, tr.h * 0.38, 0, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    const drawWindTurbines = (W: number, H: number) => {
      [[0.08, 0.53], [0.9, 0.52]].forEach(([fx, fy]) => {
        const x = fx * W;
        const y = fy * H;
        ctx.strokeStyle = 'rgba(200,220,200,0.45)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y - 55);
        ctx.stroke();
        ctx.save();
        ctx.translate(x, y - 55);
        ctx.rotate(tRef.current * 0.7);
        [0, 1, 2].forEach(i => {
          ctx.save();
          ctx.rotate(i * (Math.PI * 2 / 3));
          ctx.fillStyle = 'rgba(200,230,200,0.65)';
          ctx.beginPath();
          ctx.ellipse(0, -16, 4, 16, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        });
        ctx.restore();
        ctx.fillStyle = 'rgba(200,230,200,0.6)';
        ctx.beginPath();
        ctx.arc(x, y - 55, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    const drawRoads = (W: number, H: number) => {
      const roads = [0.67, 0.71, 0.75, 0.79];
      roads.forEach((y, i) => {
        ctx.fillStyle = '#1e1e0a';
        ctx.fillRect(0, y * H, W, H * 0.038);
        ctx.strokeStyle = 'rgba(255,255,80,0.1)';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([22, 16]);
        ctx.beginPath();
        ctx.moveTo(0, y * H + H * 0.019);
        ctx.lineTo(W, y * H + H * 0.019);
        ctx.stroke();
        ctx.setLineDash([]);
        if (mode === 'traffic') {
          const heat = ['rgba(231,76,60,0.18)', 'rgba(243,156,18,0.15)', 'rgba(243,156,18,0.12)', 'rgba(46,204,113,0.1)'][i];
          ctx.fillStyle = heat;
          ctx.fillRect(0, y * H - 2, W, H * 0.042);
        }
      });
    };

    const drawCars = (W: number, H: number) => {
      const spd = mode === 'traffic' ? 1.9 : 1;
      cars.current.forEach((c) => {
        const roads = [0.67, 0.71, 0.75, 0.79];
        const ry = roads[c.lane % 4] * H + H * 0.015;
        const dir = c.lane % 2 === 0 ? 1 : -1;
        c.x = (c.x + c.sp * spd * dir + 1) % 1;
        const cx = c.x * W;
        ctx.save();
        if (dir < 0) {
          ctx.translate(cx, ry);
          ctx.scale(-1, 1);
          ctx.translate(-cx, -ry);
        }
        ctx.fillStyle = c.col;
        ctx.beginPath();
        (ctx as any).roundRect(cx - 10, ry - 4, 20, 8, 3);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,230,80,0.9)';
        ctx.beginPath();
        ctx.ellipse(cx - 9, ry, 2, 1.5, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,80,80,0.8)';
        ctx.beginPath();
        ctx.ellipse(cx + 9, ry, 1.5, 1, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });
    };

    const drawAirParticles = (W: number, H: number) => {
      ptcls.current.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.y < 0) {
          p.x = rnd(0, 1);
          p.y = 0.95;
        }
        if (p.x < 0 || p.x > 1) p.vx *= -1;
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = p.col;
        ctx.beginPath();
        ctx.arc(p.x * W, p.y * H, p.r, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
      for (let i = 0; i < 4; i++) {
        const wave = ctx.createLinearGradient(0, H * (0.2 + i * 0.1), 0, H * (0.3 + i * 0.1));
        wave.addColorStop(0, 'rgba(52,152,219,0)');
        wave.addColorStop(0.5, `rgba(52,152,219,${0.05 + 0.02 * Math.sin(tRef.current + i)})`);
        wave.addColorStop(1, 'rgba(52,152,219,0)');
        ctx.fillStyle = wave;
        ctx.fillRect(0, H * (0.2 + i * 0.1), W, H * 0.12);
      }
    };

    const drawRain = (W: number, H: number) => {
      rain.current.forEach(r => {
        r.y += r.sp;
        if (r.y > 1) {
          r.y = 0;
          r.x = rnd(0, 1);
        }
        ctx.globalAlpha = 0.18;
        ctx.strokeStyle = '#a0c8f0';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(r.x * W, r.y * H);
        ctx.lineTo(r.x * W + 3, r.y * H + r.len);
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
    };

    const drawPowerLines = (W: number, H: number) => {
      const poles = 6;
      for (let i = 0; i < poles; i++) {
        const px = (i + 0.5) / poles * W;
        const py = H * 0.57;
        ctx.strokeStyle = 'rgba(100,110,140,0.45)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px, py - 50);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(px - 13, py - 46);
        ctx.lineTo(px + 13, py - 46);
        ctx.stroke();
        if (i < poles - 1) {
          const nx = (i + 1.5) / poles * W;
          const pulse = (Math.sin(tRef.current * 3 + i * 1.2) + 1) / 2;
          ctx.strokeStyle = `rgba(155,89,182,${0.25 + pulse * 0.45})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(px + 10, py - 46);
          ctx.quadraticCurveTo((px + nx) / 2, py - 34, nx - 10, py - 46);
          ctx.stroke();
          ctx.globalAlpha = pulse * 0.55;
          ctx.fillStyle = '#AFA9EC';
          ctx.beginPath();
          ctx.arc((px + nx) / 2, py - 34 + 6, 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }
    };

    const spawnBolt = (W: number, H: number) => {
      if (bolts.current.length < 4 && Math.random() < 0.05) {
        const segs: [number, number][] = [];
        let cx = rnd(0.1, 0.9) * W;
        let cy = rnd(0.05, 0.15) * H;
        const ey = rnd(0.25, 0.42) * H;
        segs.push([cx, cy]);
        while (cy < ey) {
          cy += rnd(10, 20);
          cx += rnd(-22, 22);
          segs.push([cx, cy]);
        }
        bolts.current.push({ segs, life: 1 });
      }
    };

    const drawStars = (W: number, H: number) => {
      ctx.save();
      for (let i = 0; i < 100; i++) {
        const x = ((Math.sin(i * 123.45) + 1) / 2) * W;
        const y = ((Math.cos(i * 543.21) + 1) / 2) * H * 0.5;
        const r = Math.random() * 1.5;
        const opacity = (Math.sin(tRef.current * 0.7 + i) + 1) / 2;
        ctx.globalAlpha = opacity * 0.6;
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    };

    const drawMoon = (W: number, H: number) => {
      const mx = W * 0.83;
      const my = H * 0.1;
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = '#d4e4f7';
      ctx.beginPath();
      ctx.arc(mx, my, 16, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = SKIES[mode][0];
      ctx.beginPath();
      ctx.arc(mx + 7, my - 4, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    };

    const drawBirds = (W: number, H: number) => {
      birds.current.forEach(b => {
        b.x = (b.x + b.sp) % 1.1;
        b.flap += 0.09;
        const bx = b.x * W;
        const by = b.y * H;
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx - 8, by);
        ctx.quadraticCurveTo(bx - 4, by + Math.sin(b.flap) * 5, bx, by);
        ctx.quadraticCurveTo(bx + 4, by + Math.sin(b.flap + Math.PI) * 5, bx + 8, by);
        ctx.stroke();
      });
    };

    const frame = () => {
      tRef.current += 0.016;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const W = canvas.width;
      const H = canvas.height;

      drawSky(W, H);
      drawStars(W, H);
      drawMoon(W, H);
      if (mode === 'weather') drawRain(W, H);
      if (mode === 'energy') {
        spawnBolt(W, H);
        drawPowerLines(W, H);
      }
      drawClouds(W, H);
      if (['home', 'traffic'].includes(mode)) drawBirds(W, H);
      drawRoads(W, H);
      drawWindTurbines(W, H);
      drawCars(W, H);
      drawTrees(W, H);
      if (mode === 'air') drawAirParticles(W, H);

      if (mode === 'energy') {
        for (let i = bolts.current.length - 1; i >= 0; i--) {
          const b = bolts.current[i];
          b.life -= 0.07;
          if (b.life <= 0) {
            bolts.current.splice(i, 1);
            continue;
          }
          ctx.globalAlpha = b.life * 0.8;
          ctx.strokeStyle = `rgba(200,190,255,${b.life})`;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          b.segs.forEach((s, j) => j === 0 ? ctx.moveTo(s[0],s[1]) : ctx.lineTo(s[0],s[1]));
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      animationFrameId = requestAnimationFrame(frame);
    };

    frame();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resize);
    };
  }, [mode]);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-[520px]" />;
};

export default CityCanvas;
