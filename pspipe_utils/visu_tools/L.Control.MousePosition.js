// Sigurd's modified version of this control. Behaves normally for normal layers,
// but for colorizable layers also prints value at mouse position

L.Control.MousePosition = L.Control.extend({
	options: {
		position: 'bottomleft',
		separator: ' : ',
		emptyString: 'Unavailable',
		lngFirst: false,
		numDigits: 5,
		valDigits: 1,
		lngFormatter: undefined,
		latFormatter: undefined,
		valFormatter: undefined,
		prefix: ""
	},

	onAdd: function (map) {
		this._container = L.DomUtil.create('div', 'leaflet-control-mouseposition');
		L.DomEvent.disableClickPropagation(this._container);
		map.on('mousemove', this._onMouseMove, this);
		this._container.innerHTML=this.options.emptyString;
		return this._container;
	},

	onRemove: function (map) {
		map.off('mousemove', this._onMouseMove)
	},

	_prev_e: null,

	_onMouseMove: function (e) {
		// Cache last mouse event so we can repaint coordinates without
		// waiting for a mouse move
		if(!e) e = this._prev_e
		this._prev_e = e;
		console.log(e);
		// Format the coordinates
		var lng = this.options.lngFormatter ? this.options.lngFormatter(e.latlng.lng) : L.Util.formatNum(e.latlng.lng, this.options.numDigits);
		var lat = this.options.latFormatter ? this.options.latFormatter(e.latlng.lat) : L.Util.formatNum(e.latlng.lat, this.options.numDigits);
		var value = this.options.lngFirst ? lng + this.options.separator + lat : lat + this.options.separator + lng;
		var prefixAndValue = this.options.prefix + ' ' + value;
		// Add map value if available
		var layer = null;
		e.target.eachLayer(function (l) {
			if(!layer && "options" in l && "colormap" in l.options)
				layer = l;
		});
		if(layer) {
			//console.log(["onMouseMove", e]);
			var vals = layer.getValueAtLayerPos(e.latlng);
			if(!(vals instanceof Array)) vals = [vals];
			for(var i = 0; i < vals.length; i++) {
				prefixAndValue += this.options.separator + (this.options.valFormatter ? this.options.valFormatter(vals[i]) : L.Util.formatNum(vals[i], this.options.valDigits));
			}
		}
		this._container.innerHTML = prefixAndValue;
	}

});

L.Map.mergeOptions({
	positionControl: false
});

L.Map.addInitHook(function () {
	if (this.options.positionControl) {
		this.positionControl = new L.Control.MousePosition();
		this.addControl(this.positionControl);
	}
});

L.control.mousePosition = function (options) {
	return new L.Control.MousePosition(options);
};
