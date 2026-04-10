

### 1. Carousel Indexing Strategy

To ensure crawlers (Googlebot, Bingbot) and LLM aggregators can ingest images typically trapped in JS-based carousels:

- **DOM Presence:** Avoid "Just-in-Time" rendering. Ensure all image elements exist in the initial HTML source or are injected into the DOM without requiring user interaction (click/swipe).
    
- **Lazy-Loading:** Use native `loading="lazy"`. Avoid legacy JS libraries that replace `src` with `data-src`, as these often fail if the bot’s execution budget is exceeded.
    
- **Fetchpriority:** Assign `fetchpriority="high"` to the first (LCP candidate) image in the carousel. Use `fetchpriority="low"` for subsequent off-screen images.
    
- **CSS Visibility:** Do not use `display: none` for inactive slides; use `opacity: 0`, `visibility: hidden`, or `absolute` positioning off-canvas. Bots frequently ignore `display: none` content.
    

---

### 2. Field-by-Field Optimization Table

|**Field**|**Recommended Length**|**Best Practices**|**SEO/AI Weight**|
|---|---|---|---|
|**File Name**|20–60 chars|Kebab-case (`fine-art-portrait.webp`). No underscores.|**High**|
|**Alt Text**|50–125 chars|Describe visual content + context. No "Image of..." prefix.|**Critical**|
|**Title Attribute**|20–50 chars|Supplemental info. Triggers tooltip.|**Low**|
|**Caption**|100–300 chars|Visible text near image. High relevance for RAG context.|**Medium/High**|
|**URL Structure**|< 128 chars|Static, clean paths. Avoid dynamic parameters.|**Medium**|

---

### 3. AI-Ready Schema.org Implementation (JSON-LD)

This structure links the visual asset to the entity, ensuring AI agents understand the relationship between the image and the product.

JSON

```
{
  "@context": "https://schema.org/",
  "@type": "Product",
  "name": "Minimalist Landscape Print",
  "image": {
    "@type": "ImageObject",
    "contentUrl": "https://cdn.example.com/art/minimalist-landscape.webp",
    "license": "https://example.com/license",
    "acquireLicensePage": "https://example.com/buy-license",
    "creator": {
      "@type": "Person",
      "name": "Jane Doe"
    },
    "description": "A high-contrast black and white photography print featuring a lone oak tree in winter, shot on 35mm film.",
    "representativeOfPage": "True"
  }
}
```

---

### 4. Advanced Metadata (IPTC & Digital Source)

For AI "Digital Source" verification and provenance, automate the injection of the following IPTC Photo Metadata fields during your ETL/Processing pipeline:

- **Digital Source Type:** Use `https://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia` if AI-generated, or `originalCapture` for authentic photography.
    
- **Web Statement of Rights:** A URL pointing to the copyright license. This triggers the "Licensable" badge in Google Images.
    
- **Credit Line:** Mandatory for brand authority across scraped datasets.
    
- **Headline/Description:** Sync these with Alt Text and Caption values to ensure metadata persists if the image is hotlinked or downloaded.
    

---

### 5. Performance & Delivery Specs

- **Formats:** Default to **AVIF** with **WebP** fallback.
    
- **Responsive Images:** Implement `srcset` with at least four breakpoints (e.g., 400w, 800w, 1200w, 1600w).
    
- **Aspect Ratio:** Use the `aspect-ratio` CSS property to reserve space and prevent Layout Shift (CLS).
    
- **CDN Headers:** Ensure `Cache-Control: public, max-age=31536000, immutable`.
    

---

### 6. Quality Matrix (Definition of Done)

- [ ] Image is present in the source HTML (View Source check).
    
- [ ] Alt text is unique and contains primary keywords.
    
- [ ] File size is < 100KB for standard web resolution.
    
- [ ] Schema.org `ImageObject` passes Google’s Rich Results Test.
    
- [ ] IPTC `WebStatementOfRights` is embedded in the binary.
    
- [ ] `fetchpriority` is applied to the LCP asset.
    
- [ ] Image URL is included in the `image-sitemap.xml`.
    

---

### Alternative Approaches Summary

- **Dynamic Rendering:** Serve a flat HTML version of the carousel to bots while maintaining the JS experience for users.
    
- **CSS-Only Carousel:** Eliminate JS entirely using `scroll-snap-type` for maximum crawlability.