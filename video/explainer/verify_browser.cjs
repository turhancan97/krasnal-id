// Point PLAYWRIGHT_MODULE at an isolated Playwright install; do not change project dependencies.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs');const assert=require('node:assert/strict');
const path=require('node:path');
process.chdir(path.resolve(__dirname,'../..'));
const qa=process.env.EXPLAINER_QA;
assert.ok(qa,'Set EXPLAINER_QA to an external screenshot directory');fs.mkdirSync(qa,{recursive:true});
const base=process.env.EXPLAINER_SITE_URL;assert.ok(base?.endsWith('/'),'Set EXPLAINER_SITE_URL to a local byte-range HTTP URL ending in /');
let browser;
(async()=>{
 browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--renderer-process-limit=2']});
 const crypto=require('node:crypto');
 const report={browser:browser.version(),video_sha256:crypto.createHash('sha256').update(fs.readFileSync('docs/media/explainer.mp4')).digest('hex'),interactions:{},screenshots:[]};
 const context=await browser.newContext({viewport:{width:1280,height:960}});const page=await context.newPage();
 const errors=[],failed=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>failed.push({url:r.url(),error:r.failure()?.errorText}));page.on('request',r=>requests.push(r.url()));
 await page.goto(base,{waitUntil:'networkidle'});await page.evaluate(()=>document.fonts.ready);
 const video=page.locator('#project-video');assert.equal(await video.count(),1);
 assert.equal(await video.getAttribute('preload'),'none');assert.equal(await video.getAttribute('autoplay'),null);assert.equal(await video.getAttribute('playsinline'),'');
 assert.ok(!requests.some(u=>u.endsWith('/explainer.mp4')),'Video fetched before playback');
 assert.ok(!requests.some(u=>u.includes('/resolve/')||u.includes('@huggingface/transformers')),'Runtime model request');
 assert.ok(await page.evaluate(()=>Boolean(document.querySelector('.teaser').compareDocumentPosition(document.querySelector('#explainer'))&Node.DOCUMENT_POSITION_FOLLOWING)));
 assert.ok(await page.evaluate(()=>Boolean(document.querySelector('#explainer').compareDocumentPosition(document.querySelector('#abstract'))&Node.DOCUMENT_POSITION_FOLLOWING)));
 await video.scrollIntoViewIfNeeded();const before=await video.boundingBox();await video.focus();await page.keyboard.press('Space');
 await page.waitForFunction(()=>document.querySelector('video').currentTime>.3);
 const media=await video.evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration,paused:v.paused}));
 assert.equal(media.width,1920);assert.equal(media.height,1080);assert.equal(media.duration,120);assert.equal(media.paused,false);
 const after=await video.boundingBox();assert.ok(Math.abs(before.height-after.height)<1&&Math.abs(before.y-after.y)<1,'Player shifted during loading');
 await page.keyboard.press('Space');await page.waitForFunction(()=>document.querySelector('video').paused);
 for(const time of [59,80,98,116]){
  await video.evaluate((v,t)=>v.currentTime=t,time);await page.waitForFunction(t=>Math.abs(document.querySelector('video').currentTime-t)<.1&&!document.querySelector('video').seeking,time);
 }
 await page.evaluate(()=>{const b=document.createElement('button');b.id='fullscreen-check';b.textContent='Fullscreen check';b.onclick=()=>document.querySelector('video').requestFullscreen();document.querySelector('video').after(b)});
 await page.locator('#fullscreen-check').click();await page.waitForFunction(()=>document.fullscreenElement===document.querySelector('video'));await page.evaluate(()=>document.exitFullscreen());await page.locator('#fullscreen-check').evaluate(b=>b.remove());
 const event=page.waitForEvent('download');await page.getByRole('link',{name:'Download video',exact:true}).click();const download=await event;assert.equal(await download.failure(),null);
 await page.getByRole('link',{name:'Transcript',exact:true}).click();await page.waitForURL('**/media/explainer-transcript.html');
 const transcript=await page.locator('main').textContent();const story=JSON.parse(fs.readFileSync('video/explainer/story.json','utf8'));
 for(const s of story.scenes.filter(s=>s.narration))assert.ok(transcript.includes(s.narration),s.id);
 await page.goto(base+'media/explainer-credits.html',{waitUntil:'networkidle'});
 const credits=JSON.parse(fs.readFileSync('video/explainer/credits.json','utf8'));
 for(const p of credits.photographs){assert.ok(await page.locator(`a[href="${p.source_url}"]`).count());assert.ok(await page.locator(`a[href="${p.license_url}"]`).count());}
 assert.ok((await page.locator('main').textContent()).includes('af_heart'));
 assert.deepEqual(errors,[]);
 assert.deepEqual(failed.filter(r=>!(r.url.endsWith('/explainer.mp4')&&r.error==='net::ERR_ABORTED')),[]);
 report.interactions={preloadNone:true,noAutoplay:true,noRuntimeModel:true,correctPlacement:true,stableHeight:true,playback:true,seeking:true,keyboard:true,fullscreen:true,download:true,transcript:true,allPhotoCredits:true,media,errors,failedRequests:failed};
 console.log(JSON.stringify(report.interactions));await context.close();
 for(const width of [1280,390])for(const colorScheme of ['light','dark']){
  const ctx=await browser.newContext({viewport:{width,height:960},colorScheme});const p=await ctx.newPage();const errs=[];p.on('pageerror',e=>errs.push(e.message));
  await p.goto(base,{waitUntil:'networkidle'});await p.evaluate(async()=>{for(const i of document.images)i.loading='eager';await document.fonts.ready;await Promise.all(Array.from(document.images).map(i=>i.decode()))});
  const dims=await p.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,videoHeight:document.querySelector('video').getBoundingClientRect().height,brokenImages:Array.from(document.images).filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src)}));
  assert.equal(dims.width,dims.scrollWidth);assert.deepEqual(dims.brokenImages,[]);assert.deepEqual(errs,[]);
  const path=`${qa}/project-${width}-${colorScheme}.png`;await p.screenshot({path,fullPage:true});
  const section=`${qa}/player-${width}-${colorScheme}.png`;await p.locator('#explainer').screenshot({path:section});
  report.screenshots.push({path,section,width,colorScheme,dimensions:dims,errors:errs});console.log(JSON.stringify(report.screenshots.at(-1)));
  if(width===390&&colorScheme==='light'){
   const v=p.locator('video');await v.focus();await p.keyboard.press('Space');await p.waitForFunction(()=>document.querySelector('video').currentTime>.1);await v.evaluate(v=>v.currentTime=54);
   await p.waitForFunction(()=>!document.querySelector('video').seeking);await p.getByRole('link',{name:'Transcript',exact:true}).focus();await p.mouse.move(0,0);await p.waitForTimeout(3000);await v.screenshot({path:`${qa}/mobile-caption-playing.png`});
  }
  for(const name of ['explainer-transcript.html','explainer-credits.html']){
   await p.goto(base+'media/'+name,{waitUntil:'networkidle'});assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth),width);
   if(width===390)await p.screenshot({path:`${qa}/${name}-${colorScheme}.png`,fullPage:true});
  }
  await ctx.close();
 }
 fs.writeFileSync(`${qa}/browser.json`,JSON.stringify(report,null,2));await browser.close();
})().catch(async e=>{console.error(e);if(browser)await browser.close();process.exitCode=1});
