L.ColorizableUtils = {
	colormaps: {
		"gray":   [[0,0x000000],[1,0xffffff]],
		"planck": [[0,0x0000ff],[0.332,0x00d7ff],[0.5,0xffedd9],[0.664,0xffb400],[0.828,0xff4b00],[1,0x640000]],
		"wmap":   [[0,0x000080],[0.15,0x0000ff],[0.4,0x00ffff],[0.7,0xffff00],[0.9,0xff5500],[1,0x800000]],
		"iron":   [[0.00000,0x000000],[0.00840,0x000024],[0.01681,0x000033],[0.02521,0x000042],[0.03361,0x000051],[0.04202,0x02005a],[0.05042,0x040063],[0.05882,0x07006a],[0.06723,0x0b0073],[0.07563,0x0e0077],[0.08403,0x14007b],[0.09244,0x1b0080],[0.10084,0x210085],[0.10924,0x290089],[0.11765,0x30008c],[0.12605,0x37008f],[0.13445,0x3d0092],[0.14286,0x420095],[0.15126,0x480096],[0.15966,0x4e0097],[0.16807,0x540098],[0.17647,0x5b0099],[0.18487,0x61009b],[0.19328,0x68009b],[0.20168,0x6e009c],[0.21008,0x73009d],[0.21849,0x7a009d],[0.22689,0x80009d],[0.23529,0x86009d],[0.24370,0x8b009d],[0.25210,0x92009c],[0.26050,0x98009b],[0.26891,0x9d009b],[0.27731,0xa2009b],[0.28571,0xa7009a],[0.29412,0xab0099],[0.30252,0xaf0198],[0.31092,0xb20197],[0.31933,0xb60295],[0.32773,0xb90495],[0.33613,0xbc0593],[0.34454,0xbf0692],[0.35294,0xc10890],[0.36134,0xc30b8e],[0.36975,0xc60d8b],[0.37815,0xc91187],[0.38655,0xcb1484],[0.39496,0xce177f],[0.40336,0xd01a79],[0.41176,0xd21d74],[0.42017,0xd4216f],[0.42857,0xd62567],[0.43697,0xd92961],[0.44538,0xdb2e59],[0.45378,0xdd314e],[0.46218,0xdf3542],[0.47059,0xe03836],[0.47899,0xe23c2a],[0.48739,0xe4401e],[0.49580,0xe54419],[0.50420,0xe74814],[0.51261,0xe84c10],[0.52101,0xea4e0c],[0.52941,0xeb520a],[0.53782,0xec5608],[0.54622,0xed5a07],[0.55462,0xee5d05],[0.56303,0xef6004],[0.57143,0xf06403],[0.57983,0xf16703],[0.58824,0xf16a02],[0.59664,0xf26d01],[0.60504,0xf37101],[0.61345,0xf47400],[0.62185,0xf47800],[0.63025,0xf57d00],[0.63866,0xf68100],[0.64706,0xf78500],[0.65546,0xf88800],[0.66387,0xf88b00],[0.67227,0xf98e00],[0.68067,0xf99100],[0.68908,0xfa9500],[0.69748,0xfb9a00],[0.70588,0xfc9f00],[0.71429,0xfda300],[0.72269,0xfda800],[0.73109,0xfdac00],[0.73950,0xfeb000],[0.74790,0xfeb300],[0.75630,0xfeb800],[0.76471,0xfebb00],[0.77311,0xfebf00],[0.78151,0xfec300],[0.78992,0xfec700],[0.79832,0xfeca01],[0.80672,0xfecd02],[0.81513,0xfed005],[0.82353,0xfed409],[0.83193,0xfed80c],[0.84034,0xffdb0f],[0.84874,0xffdd17],[0.85714,0xffe020],[0.86555,0xffe327],[0.87395,0xffe532],[0.88235,0xffe83f],[0.89076,0xffeb4b],[0.89916,0xffee58],[0.90756,0xffef66],[0.91597,0xfff174],[0.92437,0xfff286],[0.93277,0xfff495],[0.94118,0xfff5a4],[0.94958,0xfff7b3],[0.95798,0xfff8c0],[0.96639,0xfff9cb],[0.97479,0xfffbd8],[0.98319,0xfffde4],[0.99160,0xfffeef],[1.00000,0xfffff9]],
	},
	lookup_tables: {},
	apply_scale: function(vmin, vmax, scale, skew) {
		if(skew >= 1)
			return [(vmax+(vmin-vmax)/skew)*scale,vmax*scale];
		else
			return [vmin*scale, (vmin+(vmax-vmin)*skew)*scale];
	},
	colorize: function (idata, opts) {
		var opts = Object.assign({colormap:"planck", min: -1, max:1, scale:1, skew:1, nbit:8, zeronan:false}, opts);
		var buf  = new ArrayBuffer(idata.length*4);
		var rgba = new Uint32Array(buf);
		var tab  = this.get_lookup_table(opts.colormap, opts.nbit)
		var N    = 1<<opts.nbit;
		var [v1, v2] = this.apply_scale(opts.min, opts.max, opts.scale, opts.skew);
		for(var i = 0; i < idata.length; i++) {
			var val  = idata[i];
			if(opts.zeronan && isNaN(val)) val = 0;
			if(isNaN(val)) {
				rgba[i] = 0;
			} else {
				var x    = ((val-v1)/(v2-v1)*N)|0;
				if(x < 0) x = 0; if(x >= N) x = N-1;
				rgba[i] = tab[x];
			}
		}
		return new Uint8ClampedArray(buf);
	},
	build_lookup_table: function (colormap, nbit) {
		var N = 1<<nbit;
		var buf  = new ArrayBuffer(N*4);
		var rgba = new Uint8ClampedArray(buf);
		var tab  = this.colormaps[colormap];
		for(var i = 0; i < N; i++) {
			var x = i/N;
			var j;
			if(x < 0) x = 0; if(x > 1) x = 1;
			for(j = 1; j < tab.length && tab[j][0] < x; j++);
			var x1 = tab[j-1][0], y1 = tab[j-1][1];
			var x2 = tab[j-0][0], y2 = tab[j-0][1];
			var r  = (x-x1)/(x2-x1);
			rgba[4*i+2] = (y1&0xff)*(1-r)+(y2&0xff)*r;
			y1 >>>= 8; y2 >>>= 8;
			rgba[4*i+1] = (y1&0xff)*(1-r)+(y2&0xff)*r;
			y1 >>>= 8; y2 >>>= 8;
			rgba[4*i+0] = (y1&0xff)*(1-r)+(y2&0xff)*r;
			rgba[4*i+3] = 0xff;
		}
		return new Uint32Array(buf);
	},
	get_lookup_table: function(colormap, nbit) {
		if(!(colormap in this.lookup_tables)) this.lookup_tables[colormap] = {}
		if(!(nbit in this.lookup_tables[colormap]))
			this.lookup_tables[colormap][nbit] = this.build_lookup_table(colormap, nbit);
		return this.lookup_tables[colormap][nbit];
	},
	merge_rgb: function (idatas, opts) {
		// create an rgb image from up to 3 separate data arrays
		var opts = Object.assign({mins: [-1,-1,-1], maxs:[1,1,1], scales:[1,1,1], skews:[1,1,1], nbit:8}, opts);
		console.assert(idatas.length == 3, "First argument of merge_rbg must have length 3 corresponding to R, G and B. Pass null for some of these if you want to skip them. Provided object had length %d", idatas.length);
		// Find the length
		var n = 0;
		for(var ci = 0; ci < 3; ci++) {
			if(idatas[ci] == null) continue;
			if(n == 0) { n = idatas[ci].length; }
			else       { console.assert(n == idatas[ci].length, "Inconsistent length for fields in merge_rgb"); }
		}
		var buf  = new ArrayBuffer(n*4);
		var rgba = new Uint8ClampedArray(buf);
		var N    = 1<<opts.nbit;
		// We need to keep track of which tiles had no data so that we can
		// make those transparent.
		var hitbuf= new ArrayBuffer(n);
		var hits  = new Uint8ClampedArray(hitbuf);
		for(var ci = 0; ci < 3; ci++) {
			var idata = idatas[ci];
			if(idata == null) continue;
			var [v1, v2] = this.apply_scale(opts.mins[ci], opts.maxs[ci], opts.scales[ci], opts.skews[ci]);
			for(var i = 0; i < n; i++) {
				var val  = idata[i];
				if(isNaN(val)) {
					rgba[4*i+ci] = 0;
				} else {
					var x    = ((val-v1)/(v2-v1)*N)|0;
					if(x < 0) x = 0; if(x >= N) x = N-1;
					rgba[4*i+ci] = x;
					hits[i]++;
				}
			}
		}
		// Set the alpha value
		for(var i = 0; i < n; i++) {
			if(hits[i] == 0) {
				rgba[4*i+3] = 0;
			} else {
				rgba[4*i+3] = 0xff;
			}
		}
		return rgba;
	},
	decode: function(imgdata) {
		// First copy out the non-redundant values of the RGBA input we get
		var ibuf    = new ArrayBuffer(imgdata.width*imgdata.height);
		var idata   = new Uint8Array(ibuf);
		for(var i = 0; i < idata.length; i++)
			idata[i] = imgdata.data[4*i];
		// First parse the metadata
		var nbyte   = idata[0];
		// Cumbersome
		function get_as_double(idata, offset) {
			var buf = new ArrayBuffer(8);
			var arr = new Uint8Array(buf);
			for(var i = 0; i < 8; i++) arr[i] = idata[i+offset];
			return (new Float64Array(buf))[0];
		}
		var quantum = get_as_double(idata, 1);
		var width   = imgdata.width;
		var height  = ((imgdata.height-1)/nbyte)|0;
		var npix    = width*height;
		// We can now allocate our output buffer. We will use float32
		var obuf    = new ArrayBuffer(npix*4);
		var odata   = new Float32Array(obuf);
		for(var y = 0; y < height; y++) {
			for(var x = 0; x < width; x++) {
				var ipix = (y+1)*width+x;
				var opix = y*width+x;
				// Read in the full, n-byte integer in sign,mag format
				var v = 0;
				var nff = 0;
				for(var b = nbyte-1; b >= 0; b--) {
					v <<= 8;
					v |= idata[ipix+b*npix];
					nff += (v&0xff)==0xff;
				}
				if(nff==nbyte) {
					// We're masked
					odata[opix] = NaN
				} else {
					if(v&1) v = -(v>>>1);
					else    v >>>= 1;
					odata[opix] = v*quantum;
				}
			}
		}
		return {width: width, height: height, nbyte: nbyte, quantum: quantum, data:odata};
	},
};

L.TileLayer.Colorizable = L.TileLayer.extend({

	options: {
		colormap: "planck",
		min: -500,
		max:  500,
		scale:  1,
		skew:   1,
		nbit:   8,
	},

	initialize: function (url, opts) {
		L.TileLayer.prototype.initialize.call(this, url, opts);
		this.cache = null;
	},

	setColors: function (opts) {
		L.setOptions(this, opts);
		this._updateTiles();
	},

	setCache: function (cache) {
		// I want this.cache to point to the same object as cache, so can't use object.assign
		if(!("t" in cache)) cache.t = 0;
		if(!("data" in cache)) cache.data = {};
		if(!("nmax" in cache)) cache.nmax = 100;
		this.cache = cache;
	},

	url2key: function (url) { return url; },

	createTile: function (coords, done) {
		var url  = this.getTileUrl(coords);
		var tile = this._getFromCache(this.url2key(url));
		if(tile != null) {
			// Since an element can only have one parent leaflet gets confused if we
			// return the same element again and again. So return a copy instead.
			var otile = tile.cloneNode(false);
			otile.raw = tile.raw;
			otile.complete = true;
			this._updateTile(otile);
			L.Util.requestAnimFrame(L.bind(done, this, null, otile));
			return otile;
		} else {
			var img  = document.createElement("img");
			var tile = document.createElement("canvas");
			L.DomEvent.on(img, 'load',  L.bind(this._tileOnLoad,  this, done, tile, img, url));
			//L.DomEvent.on(img, 'error', L.bind(this._tileOnError, this, done, tile));
			L.DomEvent.on(img, 'error', function(a,b,c,d,e,f,g,h) {
				console.log(["createTile error", a, b, c, d, e, f, g, h]);
			});
			if (this.options.crossOrigin) { tile.crossOrigin = ''; }
			tile.alt = '';
			tile.setAttribute('role', 'presentation');
			tile.complete = false;
			img.src = url;
			return tile;
		}
	},

	_tileOnLoad: function (done, tile, img, url) {
		// First copy over the tile data
		tile.width = img.width;
		tile.height= img.height;
		var context  = tile.getContext("2d");
		// Read the image data. This will be RGBA, even though our images are grayscale.
		// So we only need one byte out of 4 later.
		context.drawImage(img, 0, 0);
		var imgdata  = context.getImageData(0, 0, img.width, img.height);
		var res      = L.ColorizableUtils.decode(imgdata);
		// Update the canvas with the real tile size
		tile.width   = res.width;
		tile.height  = res.height;
		tile.raw     = res.data;
		tile.complete= true;
		this._addToCache(this.url2key(url), tile);
		this._updateTile(tile);
		// For https://github.com/Leaflet/Leaflet/issues/3332
		if (L.Browser.ielt9) {
			setTimeout(L.bind(done, this, null, tile), 0);
		} else {
			done(null, tile);
		}
	},

	_updateTile: function (tile) {
		var rgba     = L.ColorizableUtils.colorize(tile.raw, this.options);
		var imgdata  = new ImageData(rgba, tile.width, tile.height);
		var context  = tile.getContext("2d");
		context.putImageData(imgdata, 0, 0);
	},

	_updateTiles: function () {
		if (!this._map) { return; }
		for (var key in this._tiles) {
			var tile = this._tiles[key];
			// Don't try to update an invalid tile. I'm not
			// sure why this happens - maybe it hasn't been fully
			// loaded yet. This doesn't result in missing tiles, so
			// I guess they recover. Without this check, the colorize
			// function can fail. That seems to have lead to the different
			// layers ending up in inconsistent state, e.g. with different
			// color maps etc.
			if(tile.el.raw) this._updateTile(tile.el);
		}
	},

	_addToCache: function (url, tile) {
		if(this.cache == null) return;
		if(url in this.cache.data) {
			// just mark it as recent
			this.cache.data[url].t = this.cache.t++;
		} else {
			var ncache = this.cache.data.length;
			if(ncache >= this.cache.nmax) {
				// too much in cache, remove oldest
				console.log("cache free");
				var tmin = null, umin;
				for(var u in this.cache.data) {
					if(tmin == null || u.t < tmin) {
						tmin = u.t;
						umin = u;
					}
				}
				delete this.cache.data[umin];
			}
			// then add to cache
			this.cache.data[url] = {t: this.cache.t++, tile: tile};
		}
	},

	_getFromCache: function (url) {
		if(this.cache == null || !(url in this.cache.data)) return null;
		else {
			// Mark as recent
			this.cache.data[url].t = this.cache.t++;
			return this.cache.data[url].tile;
		}
	},

	getValueAtLayerPos: function (pos) {
		var map     = this._map;
		var pix     = map.project(pos, this._tileZoom).floor();
		var tsize   = this.getTileSize();
		var tsize0  = L.GridLayer.prototype.getTileSize.call(this);
		var tcoord  = pix.unscaleBy(tsize).floor();
		var ratio   = tsize0.unscaleBy(tsize);
		var tsub    = pix.subtract(tcoord.scaleBy(tsize)).scaleBy(ratio).floor();
		tcoord.z    = this._map.getZoom();
		var key     = this._tileCoordsToKey(tcoord);
		//console.log([this, pos, this._tileZoom, pix, tsize, tsize0, tcoord, tsub, key, key in this._tiles]);
		if(!(key in this._tiles)) return Number.NaN;
		var tile  = this._tiles[key];
		//console.log(["A",tile, [tsub.x,tsub.y], [tsize0.x, tsize0.y], tsub.y*tsize0.x+tsub.x, tile.el.raw[tsub.y*tsize0.x+tsub.x]]);
		var val   = tile.el.raw[tsub.y*tsize0.x+tsub.x];
		return val;
	}

});

L.TileLayer.RGBLayer = L.TileLayer.Colorizable.extend({

	options: {
		mins: [-500,-500,-500],
		maxs: [ 500, 500, 500],
		scales: [1, 1, 1],
		skews:  [1, 1, 1],
		nbit:   8,
	},

	initialize: function (url, opts) {
		L.TileLayer.prototype.initialize.call(this, url, opts);
		this.cache = null;
	},

	getTileUrl: function(coords) {
		var data = {
			r: L.Browser.retina ? '@2x' : '',
			s: this._getSubdomain(coords),
			x: coords.x,
			y: coords.y,
			z: this._getZoomForUrl()
		};
		if (this._map && !this._map.options.crs.infinite) {
			var invertedY = this._globalTileRange.max.y - coords.y;
			if (this.options.tms) {
				data['y'] = invertedY;
			}
			data['-y'] = invertedY;
		}
		console.assert(this._url.length == 3, "RGBLayer expects 'url' to be a list of 3 url strings, one for each or R, G and b, not just a plain string");
		var urls = [];
		for(var i = 0; i < 3; i++) {
			var url = this._url[i] == null ? null : L.Util.template(this._url[i], L.extend(data, this.options));
			urls.push(url);
		}
		return urls;
	},

	url2key: function (urls) {
		var key = "";
		for(var i = 0; i < urls.length; i++) {
			if(i>0) key += "|";
			key += String(urls[i]);
		}
		return key;
	},

	createTile: function (coords, done) {
		var urls = this.getTileUrl(coords);
		var tile = this._getFromCache(this.url2key(urls));
		if(tile != null) {
			// Since an element can only have one parent leaflet gets confused if we
			// return the same element again and again. So return a copy instead.
			var otile = tile.cloneNode(false);
			otile.raw = tile.raw;
			otile.complete = true;
			this._updateTile(otile);
			L.Util.requestAnimFrame(L.bind(done, this, null, otile));
			return otile;
		} else {
			var tile = document.createElement("canvas");
			tile.alt = '';
			tile.setAttribute('role', 'presentation');
			// Count how many field we need to load before we're done
			tile.nleft = 0;
			for(var field = 0; field < 3; field++) {
				if(urls[field] != null) tile.nleft++;
			}
			// This will store the decoded image data for each component
			tile.raw = [null,null,null];
			// Set up the loading of each image
			for(var field = 0; field < 3; field++) {
				if(urls[field] == null) continue;
				// Create an image for each field
				var img  = document.createElement("img");
				L.DomEvent.on(img, 'load',  L.bind(this._tileOnLoad,  this, field, done, tile, img, urls));
				L.DomEvent.on(img, 'error', function(a,b,c,d,e,f,g,h) {
					console.log(["createTile error", a, b, c, d, e, f, g, h]);
				});
				img.src = urls[field];
			}
			//if (this.options.crossOrigin) { tile.crossOrigin = ''; }
			return tile;
		}
	},

	_tileOnLoad: function (field, done, tile, img, urls) {
		// First copy over the image data
		var canvas    = document.createElement("canvas");
		canvas.width  = img.width;
		canvas.height = img.height;
		var context   = canvas.getContext("2d");
		context.drawImage(img, 0, 0);
		var imgdata   = context.getImageData(0, 0, img.width, img.height);
		// Then decode
		var res      = L.ColorizableUtils.decode(imgdata);
		// Store the result
		tile.raw[field] = res.data;
		tile.nleft--; // potential race condition here
		// Have we loaded all the fields yet?
		if(tile.nleft > 0) return;
		// Yes, so move on to the next steps
		tile.width   = res.width;
		tile.height  = res.height;
		this._addToCache(this.url2key(urls), tile);
		this._updateTile(tile);
		// For https://github.com/Leaflet/Leaflet/issues/3332
		if (L.Browser.ielt9) {
			setTimeout(L.bind(done, this, null, tile), 0);
		} else {
			done(null, tile);
		}
	},

	_updateTile: function (tile) {
		var rgba     = L.ColorizableUtils.merge_rgb(tile.raw, this.options);
		var imgdata  = new ImageData(rgba, tile.width, tile.height);
		var context  = tile.getContext("2d");
		context.putImageData(imgdata, 0, 0);
	},

	getValueAtLayerPos: function (pos) {
		var map     = this._map;
		var pix     = map.project(pos, this._tileZoom).floor();
		var tsize   = this.getTileSize();
		var tsize0  = L.GridLayer.prototype.getTileSize.call(this);
		var tcoord  = pix.unscaleBy(tsize).floor();
		var ratio   = tsize0.unscaleBy(tsize);
		var tsub    = pix.subtract(tcoord.scaleBy(tsize)).scaleBy(ratio).floor();
		tcoord.z    = this._map.getZoom();
		var key     = this._tileCoordsToKey(tcoord);
		//console.log([this, pos, this._tileZoom, pix, tsize, tsize0, tcoord, tsub, key, key in this._tiles]);
		if(!(key in this._tiles)) return [];
		var tile  = this._tiles[key];
		//console.log(["A",tile, [tsub.x,tsub.y], [tsize0.x, tsize0.y], tsub.y*tsize0.x+tsub.x, tile.el.raw[tsub.y*tsize0.x+tsub.x]]);
		var vals  = [];
		for(var i = 0; i < tile.el.raw.length; i++)
			if(tile.el.raw[i] != null)
				vals.push(tile.el.raw[i][tsub.y*tsize0.x+tsub.x]);
		return vals;
	}

});

L.TileLayer.DiffLayer = L.TileLayer.Colorizable.extend({

	initialize: function (url, opts) {
		L.TileLayer.prototype.initialize.call(this, url, opts);
		this.cache = null;
	},

	getTileUrl: function(coords) {
		var data = {
			r: L.Browser.retina ? '@2x' : '',
			s: this._getSubdomain(coords),
			x: coords.x,
			y: coords.y,
			z: this._getZoomForUrl()
		};
		if (this._map && !this._map.options.crs.infinite) {
			var invertedY = this._globalTileRange.max.y - coords.y;
			if (this.options.tms) {
				data['y'] = invertedY;
			}
			data['-y'] = invertedY;
		}
		console.assert(this._url.length == 2, "DiffLayer expects 'url' to be a list of 2 url strings to be subtracted, not just a plain string");
		var urls = [];
		for(var i = 0; i < 2; i++) {
			var url = this._url[i] == null ? null : L.Util.template(this._url[i], L.extend(data, this.options));
			urls.push(url);
		}
		return urls;
	},

	url2key: function (urls) {
		var key = "";
		for(var i = 0; i < urls.length; i++) {
			if(i>0) key += "|";
			key += String(urls[i]);
		}
		return key;
	},

	createTile: function (coords, done) {
		var urls = this.getTileUrl(coords);
		var tile = this._getFromCache(this.url2key(urls));
		if(tile != null) {
			// Since an element can only have one parent leaflet gets confused if we
			// return the same element again and again. So return a copy instead.
			var otile = tile.cloneNode(false);
			otile.raw = tile.raw;
			otile.complete = true;
			this._updateTile(otile);
			L.Util.requestAnimFrame(L.bind(done, this, null, otile));
			return otile;
		} else {
			var tile = document.createElement("canvas");
			tile.alt = '';
			tile.setAttribute('role', 'presentation');
			// Count how many field we need to load before we're done
			tile.nleft = 0;
			for(var field = 0; field < 2; field++) {
				if(urls[field] != null) tile.nleft++;
			}
			// This will store the decoded image data for each component
			tile.raw = [null,null];
			// Set up the loading of each image
			for(var field = 0; field < 2; field++) {
				if(urls[field] == null) continue;
				// Create an image for each field
				var img  = document.createElement("img");
				L.DomEvent.on(img, 'load',  L.bind(this._tileOnLoad,  this, field, done, tile, img, urls));
				L.DomEvent.on(img, 'error', function(a,b,c,d,e,f,g,h) {
					console.log(["createTile error", a, b, c, d, e, f, g, h]);
				});
				img.src = urls[field];
			}
			//if (this.options.crossOrigin) { tile.crossOrigin = ''; }
			return tile;
		}
	},

	_tileOnLoad: function (field, done, tile, img, urls) {
		// First copy over the image data
		var canvas    = document.createElement("canvas");
		canvas.width  = img.width;
		canvas.height = img.height;
		var context   = canvas.getContext("2d");
		context.drawImage(img, 0, 0);
		var imgdata   = context.getImageData(0, 0, img.width, img.height);
		// Then decode
		var res      = L.ColorizableUtils.decode(imgdata);
		// Store the result
		tile.raw[field] = res.data;
		tile.nleft--; // potential race condition here
		// Have we loaded all the fields yet?
		if(tile.nleft > 0) return;
		// Yes, so move on to the next steps
		tile.width   = res.width;
		tile.height  = res.height;
		this._addToCache(this.url2key(urls), tile);
		this._updateTile(tile);
		// For https://github.com/Leaflet/Leaflet/issues/3332
		if (L.Browser.ielt9) {
			setTimeout(L.bind(done, this, null, tile), 0);
		} else {
			done(null, tile);
		}
	},

	_updateTile: function (tile) {
		// Get the difference between the two
		var diff    = new Float32Array(tile.raw[0].length);
		for(var i = 0; i < diff.length; i++)
			diff[i] = tile.raw[1][i]-tile.raw[0][i];
		var rgba     = L.ColorizableUtils.colorize(diff, this.options);
		var imgdata  = new ImageData(rgba, tile.width, tile.height);
		var context  = tile.getContext("2d");
		context.putImageData(imgdata, 0, 0);
	},

	getValueAtLayerPos: function (pos) {
		var map     = this._map;
		var pix     = map.project(pos, this._tileZoom).floor();
		var tsize   = this.getTileSize();
		var tsize0  = L.GridLayer.prototype.getTileSize.call(this);
		var tcoord  = pix.unscaleBy(tsize).floor();
		var ratio   = tsize0.unscaleBy(tsize);
		var tsub    = pix.subtract(tcoord.scaleBy(tsize)).scaleBy(ratio).floor();
		tcoord.z    = this._map.getZoom();
		var key     = this._tileCoordsToKey(tcoord);
		//console.log([this, pos, this._tileZoom, pix, tsize, tsize0, tcoord, tsub, key, key in this._tiles]);
		if(!(key in this._tiles)) return [];
		var tile  = this._tiles[key];
		//console.log(["A",tile, [tsub.x,tsub.y], [tsize0.x, tsize0.y], tsub.y*tsize0.x+tsub.x, tile.el.raw[tsub.y*tsize0.x+tsub.x]]);
		var vals  = [];
		for(var i = 0; i < tile.el.raw.length; i++)
			if(tile.el.raw[i] != null)
				vals.push(tile.el.raw[i][tsub.y*tsize0.x+tsub.x]);
		if(vals.length > 1) vals.push(vals[1]-vals[0]);
		return vals;
	}

});



L.tileLayer.colorizable = function(url, options) {
	return new L.TileLayer.Colorizable(url, options);
};

L.tileLayer.rgblayer = function(url, options) {
	return new L.TileLayer.RGBLayer(url, options);
};

L.tileLayer.difflayer = function(url, options) {
	return new L.TileLayer.DiffLayer(url, options);
};
