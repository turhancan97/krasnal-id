import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrate, closeCallStatistics, selectExamples, enrichMetadata } from './metadata.mjs';
import { rankStatues, weightByLocation, distanceMetres } from '../retrieval.mjs';

test('leave-one-out excludes its own image, collapses references to statues, and preserves cosine ties', () => {
  const images = [{ id:'a1', dwarf:'a' }, { id:'b1', dwarf:'b' }, { id:'a2', dwarf:'a' }, { id:'b2', dwarf:'b' }];
  const vectors = [[1,0],[1,0],[0.8,0.6],[0,1]];
  const ranked = rankStatues(vectors[0], vectors, images, 0);
  assert.deepEqual(ranked.map((h)=>h.image.id), ['b1','a2']);
  const result = calibrate(vectors, images);
  assert.equal(result.measured.folds, 4);
  assert.equal(result.queries[0].correct, false);
});

test('weight zero is exact, missing positions get no bonus, and weighted ties keep cosine order', () => {
  const ranked = [{image:{dwarf:'a'},cosine:0.5,score:0.5},{image:{dwarf:'b'},cosine:0.5,score:0.5}];
  const dwarfs = new Map([['a',{lat:51,lon:17}]]);
  const position = {lat:51,lon:17};
  const prior = {kernel:'exponential',scale_m:200,weight:0};
  assert.deepEqual(weightByLocation(ranked,dwarfs,position,prior),ranked);
  const weighted = weightByLocation(ranked,dwarfs,position,{...prior,weight:0.2});
  assert.equal(weighted[0].score,0.7);assert.equal(weighted[1].score,0.5);
  assert.equal(distanceMetres(position,position),0);
  assert.ok(Math.abs(distanceMetres({lat:0,lon:0},{lat:0,lon:1})-111195)<1);
  assert.deepEqual(weightByLocation(ranked,new Map(),position,{...prior,weight:0.2}).map((h)=>h.image.dwarf),['a','b']);
});

test('close-call scan includes complete ties and finds the last valid boundary after an accuracy rebound', () => {
  const queries = [false,false,true,true,false,false,true].map((correct,i)=>({id:`id${i}`,margin:i,correct}));
  // Prefix first reaches 50%, then drops below it again. Last valid cut is 6.
  assert.equal(closeCallStatistics(queries).threshold,6);
  const tied = [{id:'a',margin:0,correct:false},{id:'b',margin:0,correct:true},{id:'c',margin:0,correct:false},{id:'d',margin:1,correct:true}];
  const result=closeCallStatistics(tied);assert.equal(result.threshold,1);assert.equal(result.share_of_queries,3/4);
  assert.equal(result.top_1_below,1/3);assert.equal(result.top_1_above,1);
  assert.throws(()=>closeCallStatistics([{id:'a',margin:0,correct:true},{id:'b',margin:1,correct:true}]));
});

test('examples use distinct statues and exclude the pillar family from percentile choices', () => {
  const queries=Array.from({length:50},(_,i)=>({id:`img${String(i).padStart(2,'0')}`,dwarf:`d${i%9}`,margin:i,correct:true}));
  queries.push({id:'commons-4074360',dwarf:'pillar',margin:0,correct:true});
  const dwarfs=[{id:'pillar',name:'Słupniki Solne'},...Array.from({length:9},(_,i)=>({id:`d${i}`,name:`Statue ${i}`}))];
  const examples=selectExamples(queries,dwarfs).image_ids;
  assert.equal(examples[0],'commons-4074360');assert.equal(examples.length,6);
  assert.equal(new Set(examples.map((id)=>queries.find((q)=>q.id===id).dwarf)).size,6);
  assert.deepEqual(selectExamples([...queries].reverse(),dwarfs).image_ids,examples);
});

test('metadata refuses shuffled or relabelled image rows before reading any vectors', () => {
  const manifest={staging_sha256:'hash',images:[{image_id:'a',dwarf_id:'a'},{image_id:'b',dwarf_id:'b'}]};
  const metadata={staging_sha256:'hash',images:[{id:'b',dwarf:'b'},{id:'a',dwarf:'a'}]};
  assert.throws(()=>enrichMetadata(metadata,manifest,'/missing','/missing'),/image order/);
  metadata.images=[{id:'a',dwarf:'wrong'},{id:'b',dwarf:'b'}];
  assert.throws(()=>enrichMetadata(metadata,manifest,'/missing','/missing'),/labels/);
});
