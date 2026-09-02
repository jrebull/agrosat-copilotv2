// useMap: imperative MapLibre lifecycle extracted from MapCanvas.vue (US-058 fe/A).
//
// Owns everything the component should NOT carry inline:
//   - init: dynamic import of maplibre-gl + CSS, `new Map`, Navigation/Scale
//     controls, basemap style (via useBasemap.buildBasemapStyle)
//   - overlay sources/layers: findings (fill/line/highlight) + active AOI
//     (fill/line), with stable layer ids
//   - sync: findings <- chat store, AOI <- map store (setData + parcel count +
//     fit-to-features)
//   - interactions: parcel click (i18n popup + optional onParcelSelect hook),
//     hover-highlight, cursor coords, rectangle-draw (down/move/up + Escape)
//   - flyTo helpers: demo AOI, locate a single parcel
//   - live basemap switch (setStyle wipes layers -> re-add overlays on styledata)
//   - destroy: map.remove() + listener/ref reset
//
// What stays in MapCanvas: the <template>, the rubber-band visual (it reads the
// `drawRect` ref returned here), the chips (MapDrawToolbar/MapCropLegend/
// MapCoordsReadout), defineExpose and the mapApiRef registration (layout glue).
//
// SSR-safe: `import("maplibre-gl")` lives INSIDE initMap (the component calls it
// under import.meta.client). No module-level singletons — each useMap() call
// owns its own map instance, so multi-instance and per-navigation cleanup work.

// vue-tsc 3.3 ya no expone el namespace global `GeoJSON`; se importa el tipo explicitamente.
import type * as GeoJSON from "geojson";
import { storeToRefs } from "pinia";
import type {
  Map as MlMap,
  GeoJSONSource,
  MapGeoJSONFeature,
  MapMouseEvent,
} from "maplibre-gl";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";
import type { Finding } from "~/types/agent";
import type { AoiPolygon, LngLat } from "~/types/map";
import { colorForCrop, colorForDemo } from "~/utils/cropPalette";
import { buildBasemapStyle } from "~/composables/useBasemap";
import { loadPredictionParcels } from "~/utils/demoPreview";

const FINDINGS_SOURCE = "findings";
const FINDINGS_FILL = "findings-fill";
const FINDINGS_LINE = "findings-line";
const FINDINGS_HL = "findings-highlight";
const AOI_SOURCE = "active-aoi";
const AOI_FILL = "active-aoi-fill";
const AOI_LINE = "active-aoi-line";

export interface UseMapOptions {
  /**
   * Called when a parcel is clicked, with its (real) parcel_id and feature
   * properties. The cross-store link (map-store highlight + chat-store
   * activeParcelId) is wired by the caller (MapCanvas), keeping useMap
   * agnostic/testable. fe/B plugs the chat link here.
   */
  onParcelSelect?: (parcelId: number, props: Record<string, unknown>) => void;
}

export interface UseMapHandle {
  /** Create the map in `container` (call under import.meta.client). */
  initMap: (container: HTMLElement) => Promise<void>;
  /** Tear the map down (call in onBeforeUnmount). */
  destroyMap: () => void;
  flyToDemoAoi: () => void;
  locateParcel: (parcelId: number) => void;
  /** True once the map is loaded and overlay layers are present. */
  isReady: Ref<boolean>;
  /** Screen-space rubber-band rect while drawing; null when idle. The
   *  visual lives in MapCanvas, which reads this ref. */
  drawRect: Ref<{ x: number; y: number; w: number; h: number } | null>;
}

export function useMap(opts: UseMapOptions = {}): UseMapHandle {
  const chatStore = useChatStore();
  const mapStore = useMapStore();
  const { findings } = storeToRefs(chatStore);
  const { basemap, drawMode, parcelsVisible, activeAoi, demoView } =
    storeToRefs(mapStore);
  const { selectDrawnAoi, rectToPolygon } = useAoi();
  const { t } = useI18n();

  const isReady = ref(false);
  const drawRect = ref<{ x: number; y: number; w: number; h: number } | null>(null);

  let map: MlMap | null = null;
  let maplibre: typeof import("maplibre-gl") | null = null;
  let popupCtor: typeof import("maplibre-gl").Popup | null = null;
  let hovered: string | number | null = null;
  // Rectangle draw state (screen-space).
  let drawStart: { lng: number; lat: number; px: number; py: number } | null = null;
  // Keep watch stop-handles so destroyMap fully detaches this instance.
  const stopHandles: Array<() => void> = [];

  function findingsToGeoJSON(items: Finding[]): GeoJSON.FeatureCollection {
    const features: GeoJSON.Feature[] = [];
    // In the prediction demo, colour by the active view (predicted / true /
    // hits-errors); otherwise keep the plain per-crop colour for live data.
    const prediction = chatStore.hasPrediction;
    for (const f of items) {
      const geometry = (f as unknown as { geometry?: GeoJSON.Geometry }).geometry;
      if (!geometry) continue;
      const color = prediction
        ? colorForDemo(demoView.value, f.crop_class, f.true_class, f.correct)
        : colorForCrop(f.crop_class);
      features.push({
        type: "Feature",
        id: f.parcel_id,
        geometry,
        properties: {
          parcel_id: f.parcel_id,
          crop_class: f.crop_class ?? null,
          true_class: f.true_class ?? null,
          correct: f.correct ?? null,
          confidence: f.confidence ?? null,
          area_ha: f.area_ha ?? null,
          ndvi_mean: f.ndvi_mean ?? null,
          source: f.citation?.source ?? null,
          color,
        },
      });
    }
    return { type: "FeatureCollection", features };
  }

  function aoiFeatureCollection(geom: AoiPolygon | null): GeoJSON.FeatureCollection {
    if (!geom) return { type: "FeatureCollection", features: [] };
    return {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: geom, properties: {} }],
    };
  }

  function fitToFeatures(data: GeoJSON.FeatureCollection) {
    if (!map || !maplibre) return;
    const bounds = new maplibre.LngLatBounds();
    let any = false;
    const visit = (coords: GeoJSON.Position[]) => {
      for (const pos of coords) {
        const x = pos[0];
        const y = pos[1];
        if (x === undefined || y === undefined) continue;
        bounds.extend([x, y]);
        any = true;
      }
    };
    for (const feat of data.features) {
      const g = feat.geometry;
      if (g.type === "Polygon") g.coordinates.forEach(visit);
      else if (g.type === "MultiPolygon") g.coordinates.forEach((p) => p.forEach(visit));
    }
    if (any) map.fitBounds(bounds, { padding: 60, maxZoom: 14, duration: 600 });
  }

  function syncFindings(fit = true) {
    if (!map || !isReady.value) return;
    const data = findingsToGeoJSON(findings.value);
    const source = map.getSource(FINDINGS_SOURCE) as GeoJSONSource | undefined;
    if (source) source.setData(data);
    mapStore.setParcelCount(data.features.length);
    if (fit && data.features.length > 0) fitToFeatures(data);
  }

  function syncAoi() {
    if (!map || !isReady.value) return;
    const source = map.getSource(AOI_SOURCE) as GeoJSONSource | undefined;
    if (source) source.setData(aoiFeatureCollection(activeAoi.value?.geometry ?? null));
  }

  function addOverlayLayers() {
    if (!map) return;
    map.addSource(FINDINGS_SOURCE, {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
    });
    map.addLayer({
      id: FINDINGS_FILL,
      type: "fill",
      source: FINDINGS_SOURCE,
      paint: { "fill-color": ["get", "color"], "fill-opacity": 0.45 },
    });
    map.addLayer({
      id: FINDINGS_HL,
      type: "fill",
      source: FINDINGS_SOURCE,
      paint: { "fill-color": ["get", "color"], "fill-opacity": 0.7 },
      filter: ["==", ["get", "parcel_id"], -1],
    });
    map.addLayer({
      id: FINDINGS_LINE,
      type: "line",
      source: FINDINGS_SOURCE,
      paint: { "line-color": ["get", "color"], "line-width": 1.5 },
    });

    map.addSource(AOI_SOURCE, {
      type: "geojson",
      data: { type: "FeatureCollection", features: [] },
    });
    map.addLayer({
      id: AOI_FILL,
      type: "fill",
      source: AOI_SOURCE,
      paint: { "fill-color": "#d97706", "fill-opacity": 0.12 },
    });
    map.addLayer({
      id: AOI_LINE,
      type: "line",
      source: AOI_SOURCE,
      paint: { "line-color": "#d97706", "line-width": 2, "line-dasharray": [2, 1.5] },
    });
  }

  function wireParcelInteractions() {
    if (!map) return;
    map.on("click", FINDINGS_FILL, (e) => {
      const feature = e.features?.[0] as MapGeoJSONFeature | undefined;
      if (!feature || !map || !popupCtor) return;
      const p = feature.properties ?? {};
      const conf = p.confidence != null ? `${Math.round(Number(p.confidence) * 100)}%` : "—";
      const ndvi = p.ndvi_mean != null ? Number(p.ndvi_mean).toFixed(2) : "—";
      const area = p.area_ha != null ? `${Number(p.area_ha).toFixed(1)} ha` : "—";
      // Prediction demo: show predicted vs true crop and a hit/error mark.
      const hasTruth = p.true_class != null;
      const cropLine = hasTruth
        ? `${t("map.predicted")}: <strong>${p.crop_class ?? "—"}</strong><br/>
           ${t("map.true_crop")}: ${p.true_class}
           ${p.correct ? "✓" : "✗"}<br/>`
        : `${t("map.crop")}: ${p.crop_class ?? "—"}<br/>`;
      const html = `
        <div style="font-size:12px;line-height:1.5;min-width:150px">
          <strong>${t("chat.parcel")} ${p.parcel_id ?? "—"}</strong><br/>
          ${cropLine}
          ${t("map.confidence")}: <span style="font-variant-numeric:tabular-nums">${conf}</span><br/>
          ${t("map.ndvi")}: <span style="font-variant-numeric:tabular-nums">${ndvi}</span><br/>
          ${t("map.area")}: <span style="font-variant-numeric:tabular-nums">${area}</span><br/>
          <em>${t("map.source")}: ${p.source ?? "—"}</em>
        </div>`;
      new popupCtor({ closeButton: true }).setLngLat(e.lngLat).setHTML(html).addTo(map);

      // Cross-store link is decided by the caller (MapCanvas wires onParcelSelect
      // to map-store highlight + chat-store activeParcelId). parcel_id is REAL.
      const parcelId = p.parcel_id != null ? Number(p.parcel_id) : null;
      if (parcelId != null && Number.isFinite(parcelId)) {
        opts.onParcelSelect?.(parcelId, p as Record<string, unknown>);
      }
    });

    map.on("mousemove", FINDINGS_FILL, (e) => {
      if (!map || drawMode.value) return;
      map.getCanvas().style.cursor = "pointer";
      const id = e.features?.[0]?.properties?.parcel_id;
      if (id != null && id !== hovered) {
        hovered = id;
        map.setFilter(FINDINGS_HL, ["==", ["get", "parcel_id"], id]);
      }
    });
    map.on("mouseleave", FINDINGS_FILL, () => {
      if (!map || drawMode.value) return;
      map.getCanvas().style.cursor = "";
      hovered = null;
      map.setFilter(FINDINGS_HL, ["==", ["get", "parcel_id"], -1]);
    });

    map.on("mousemove", (e) => {
      mapStore.setCursorCoords({ lng: e.lngLat.lng, lat: e.lngLat.lat });
    });

    // Track the visible extent (US-058 AC "bbox visible"); kept for future
    // spatial scoping. getBounds().toArray() -> [[w,s],[e,n]]; flatten to
    // [minLng, minLat, maxLng, maxLat].
    map.on("moveend", () => {
      if (!map) return;
      const [[w, s], [e, n]] = map.getBounds().toArray();
      mapStore.setVisibleBbox([w, s, e, n]);
    });
  }

  // --- Rectangle draw -----------------------------------------------------
  function onCanvasDown(e: MapMouseEvent) {
    if (!drawMode.value || !map) return;
    e.preventDefault();
    drawStart = { lng: e.lngLat.lng, lat: e.lngLat.lat, px: e.point.x, py: e.point.y };
    drawRect.value = { x: e.point.x, y: e.point.y, w: 0, h: 0 };
    map.dragPan.disable();
  }
  function onCanvasMove(e: MapMouseEvent) {
    if (!drawStart) return;
    const x = Math.min(drawStart.px, e.point.x);
    const y = Math.min(drawStart.py, e.point.y);
    drawRect.value = {
      x,
      y,
      w: Math.abs(e.point.x - drawStart.px),
      h: Math.abs(e.point.y - drawStart.py),
    };
  }
  function onCanvasUp(e: MapMouseEvent) {
    if (!drawStart || !map) return;
    const start: LngLat = { lng: drawStart.lng, lat: drawStart.lat };
    const end: LngLat = { lng: e.lngLat.lng, lat: e.lngLat.lat };
    const dragged = drawRect.value && (drawRect.value.w > 6 || drawRect.value.h > 6);
    drawStart = null;
    drawRect.value = null;
    map.dragPan.enable();
    if (!dragged) {
      mapStore.setDrawMode(false);
      return;
    }
    const polygon = rectToPolygon(start, end);
    // selectDrawnAoi is synchronous (no /aois persistence): the polygon lives in
    // the map store and is sent inline as `aoi` on the next POST /chat.
    selectDrawnAoi(polygon, t("aoi.drawn_label"));
  }

  function onKeydown(ev: KeyboardEvent) {
    if (ev.key === "Escape" && drawMode.value) {
      drawStart = null;
      drawRect.value = null;
      if (map) map.dragPan.enable();
      mapStore.setDrawMode(false);
    }
  }

  // --- Public flyTo helpers ----------------------------------------------
  async function flyToDemoAoi() {
    if (!map) return;
    // Load REAL PASTIS-R parcels carrying the MODEL'S PREDICTION (out-of-sample
    // fold) and paint them; the legend toggles predicted / true / hits-errors.
    // Replaces the old synthetic Tuscany rectangles (no meaning).
    const real = await loadPredictionParcels();
    if (!real || !map) return;
    mapStore.setActiveAoi({
      id: -1,
      label: t("tools.demo"),
      area_ha: null,
      geometry: real.aoiPolygon,
    });
    mapStore.setPreviewActive(true);
    mapStore.setDemoView("pred");
    mapStore.setPredictionAccuracy(real.accuracy);
    chatStore.loadDemoParcels(real.findings);
    const [minLng, minLat, maxLng, maxLat] = real.bbox;
    map.fitBounds(
      [
        [minLng, minLat],
        [maxLng, maxLat],
      ],
      { padding: 60, duration: 900, maxZoom: 16 },
    );
  }
  function locateParcel(parcelId: number) {
    if (!map) return;
    const f = findings.value.find((x) => x.parcel_id === parcelId);
    const geom = (f as unknown as { geometry?: GeoJSON.Geometry } | undefined)?.geometry;
    if (!geom) return;
    fitToFeatures({
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: geom, properties: {} }],
    });
    if (map) map.setFilter(FINDINGS_HL, ["==", ["get", "parcel_id"], parcelId]);
  }

  // --- Lifecycle ----------------------------------------------------------
  async function initMap(container: HTMLElement) {
    if (map) return;
    maplibre = await import("maplibre-gl");
    await import("maplibre-gl/dist/maplibre-gl.css");
    popupCtor = maplibre.Popup;

    map = new maplibre.Map({
      container,
      style: buildBasemapStyle(basemap.value),
      center: [11.105, 43.305],
      zoom: 6,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibre.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibre.ScaleControl({ unit: "metric" }), "bottom-left");

    map.on("load", () => {
      if (!map) return;
      addOverlayLayers();
      wireParcelInteractions();
      map.on("mousedown", onCanvasDown);
      map.on("mousemove", onCanvasMove);
      map.on("mouseup", onCanvasUp);
      isReady.value = true;
      syncFindings();
      syncAoi();
    });

    window.addEventListener("keydown", onKeydown);

    // Live basemap switch: setStyle wipes layers, so re-add overlays on styledata.
    stopHandles.push(
      watch(basemap, (id) => {
        if (!map) return;
        isReady.value = false;
        map.setStyle(buildBasemapStyle(id));
        map.once("styledata", () => {
          if (!map) return;
          addOverlayLayers();
          wireParcelInteractions();
          isReady.value = true;
          syncFindings(false);
          syncAoi();
        });
      }),
    );
    stopHandles.push(watch(findings, () => syncFindings(), { deep: true }));
    // Re-colour (no re-fit) when the prediction demo view toggles.
    stopHandles.push(watch(demoView, () => syncFindings(false)));
    stopHandles.push(watch(activeAoi, () => syncAoi(), { deep: true }));
    stopHandles.push(
      watch(parcelsVisible, (v) => {
        if (!map || !isReady.value) return;
        const vis = v ? "visible" : "none";
        for (const id of [FINDINGS_FILL, FINDINGS_LINE, FINDINGS_HL]) {
          if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis);
        }
      }),
    );
    stopHandles.push(
      watch(drawMode, (on) => {
        if (!map) return;
        map.getCanvas().style.cursor = on ? "crosshair" : "";
      }),
    );
  }

  function destroyMap() {
    if (import.meta.client) window.removeEventListener("keydown", onKeydown);
    for (const stop of stopHandles.splice(0)) stop();
    if (map) {
      map.remove();
      map = null;
    }
    isReady.value = false;
    drawRect.value = null;
    drawStart = null;
    hovered = null;
    popupCtor = null;
    maplibre = null;
  }

  return { initMap, destroyMap, flyToDemoAoi, locateParcel, isReady, drawRect };
}
