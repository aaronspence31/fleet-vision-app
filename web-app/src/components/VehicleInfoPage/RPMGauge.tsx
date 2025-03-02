"use client";
import { useEffect, useRef } from "react";
import * as d3 from "d3";

const DEFAULT_REFRESH_RATE = 1000; // Assuming this value

// Helper functions
const degToRad = (deg: number): number => {
  return (deg * Math.PI) / 180;
};

const scale = (input: number, max: number): number => {
  return input / max;
};

interface RpmGaugeProps {
  value: number;
}

const RpmGauge = ({ value = 0 }: RpmGaugeProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const needleRef = useRef<SVGPathElement | null>(null);
  const mountedRef = useRef(false);

  // Generate the gauge on component mount (only once)
  useEffect(() => {
    if (containerRef.current && !mountedRef.current) {
      // Clear any existing content first to prevent duplicates
      containerRef.current.innerHTML = "";
      generate();
      mountedRef.current = true;
    }

    // Cleanup function to handle component unmounting
    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
      mountedRef.current = false;
    };
  }, []);

  // Update values when props change
  useEffect(() => {
    if (needleRef.current) {
      setValue(value, DEFAULT_REFRESH_RATE);
    }
  }, [value]);

  const generate = (): void => {
    if (!containerRef.current) return;

    const el = containerRef.current;
    const svg = d3
      .select(el)
      .append("svg")
      .attr("width", "100%")
      .attr("height", "100%");
    const g = svg.append("g").attr("transform", `translate(200, 200)`);
    const colors = [
      "#D1D1D1",
      "#AFAFAF",
      "#FFFFFF",
      "#FD3104",
      "#171717",
      "#0A0A0A",
    ];
    const ticksData = [
      { value: 0 },
      { value: 10 },
      { value: 20 },
      { value: 30 },
      { value: 40 },
      { value: 50 },
      { value: 60 },
      { value: 70 },
      { value: 80 },
    ];
    const r = 200; // width / 2

    // gradients
    const defs = svg.append("defs");

    const gradient = defs
      .append("linearGradient")
      .attr("id", "gradient1")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "50%")
      .attr("y2", "100%");
    gradient
      .append("stop")
      .attr("offset", "50%")
      .attr("stop-color", colors[4])
      .attr("stop-opacity", 1);
    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", colors[5])
      .attr("stop-opacity", 1);

    // outer circle
    const outerRadius = r - 10;
    const innerRadius = 0;

    const circle = d3
      .arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .startAngle(0)
      .endAngle(2 * Math.PI);

    g.append("path")
      .attr("d", circle as any)
      .attr("fill", "url(#gradient1)")
      .attr("stroke", colors[1])
      .attr("stroke-width", "7");

    // ticks
    const lg = svg
      .append("g")
      .attr("class", "label")
      .attr("transform", `translate(${r}, ${r})`);
    const minAngle = -160;
    const maxAngle = 90;
    const angleRange = maxAngle - minAngle;

    const ticks = ticksData
      .reduce((acc, curr) => {
        if (curr.value === 0) {
          return acc;
        } else {
          return acc.concat(d3.range(curr.value - 10, curr.value + 10));
        }
      }, [] as number[])
      .filter((d: number) => d % 2 === 0 && d <= 80);

    lg.selectAll("line")
      .data(ticks)
      .enter()
      .append("line")
      .attr("class", "tickline")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", (d: number) => (d % 5 === 0 ? "12" : "7"))
      .attr("transform", (d: number) => {
        const scaleFunc = d3.scaleLinear().range([0, 1]).domain([0, 80]);
        const ratio = scaleFunc(d);
        const newAngle = minAngle + ratio * angleRange;
        const deviation = d % 5 === 0 ? 12 : 17;
        return `rotate(${newAngle}) translate(0, ${deviation - r})`;
      })
      .style("stroke", (d: number) => (d >= 70 ? colors[3] : colors[2]))
      .style("stroke-width", (d: number) => (d % 5 === 0 ? "3" : "1"));

    // tick texts
    lg.selectAll("text")
      .data(ticksData)
      .enter()
      .append("text")
      .attr("transform", (d: { value: number }) => {
        const scaleFunc = d3.scaleLinear().range([0, 1]).domain([0, 80]);
        const ratio = scaleFunc(d.value);
        const newAngle = degToRad(minAngle + ratio * angleRange);
        const y = (55 - r) * Math.cos(newAngle);
        const x = -1 * (52 - r) * Math.sin(newAngle);
        return `translate(${x}, ${y + 7})`;
      })
      .text((d: { value: number }) => (d.value !== 0 ? d.value / 10 : ""))
      .attr("fill", (d: { value: number }) =>
        d.value >= 70 ? colors[3] : colors[2]
      )
      .attr("font-size", "30")
      .attr("text-anchor", "middle");

    // needle
    const pointerHeadLength = r * 0.88;
    const lineData = [
      [0, -pointerHeadLength],
      [0, 15],
    ];
    const needleLine = d3.line();
    const ng = svg
      .append("g")
      .data([lineData])
      .attr("class", "pointer")
      .attr("stroke", colors[3])
      .attr("stroke-width", "6")
      .attr("stroke-linecap", "round")
      .attr("transform", `translate(${r}, ${r})`)
      .attr("z-index", "1");

    const needle = ng
      .append("path")
      .attr("d", needleLine as any)
      .attr("transform", `rotate(${-160})`);
    needleRef.current = needle.node();

    // inner circle
    const tg = svg.append("g").attr("transform", `translate(${r}, ${r})`);

    const innerArcOuterRadius = r - 80;
    const innerArcInnerRadius = 0;

    const innerArc = d3
      .arc()
      .innerRadius(innerArcInnerRadius)
      .outerRadius(innerArcOuterRadius)
      .startAngle(0)
      .endAngle(2 * Math.PI);

    tg.append("path")
      .attr("d", innerArc as any)
      .attr("stroke", colors[0])
      .attr("stroke-width", "2")
      .attr("fill", "url(#gradient1)")
      .attr("z-index", "10");

    // big text in center
    const rpmText = tg
      .append("text")
      .attr("id", "gauge-value-text") // ID for easy updates
      .text((value / 1000).toFixed(1)) // Convert to thousands and format
      .attr("font-size", "80")
      .attr("text-anchor", "middle")
      .attr("fill", colors[2])
      .attr("x", "0")
      .attr("y", "25px")
      .style("position", "absolute")
      .style("z-index", "10");

    // rpm x 1000 text
    tg.append("text")
      .text("1/min x 1000")
      .attr("font-size", "14")
      .attr("text-anchor", "middle")
      .attr("fill", colors[2])
      .attr("x", "0")
      .attr("y", "85px")
      .style("position", "absolute")
      .style("z-index", "10");

    // In Next.js, we need to handle image paths differently
    // Consider placing these images in the public folder and referencing them as:

    // lights icon
    tg.append("image")
      .attr("xlink:href", "/images/lights.svg") // Note the change from xlink:xlink:href to xlink:href
      .attr("x", "10px")
      .attr("y", "134px")
      .attr("width", "35px")
      .attr("height", "35px");

    // seat belt icon
    tg.append("image")
      .attr("xlink:href", "/images/seat-belt.svg")
      .attr("x", "56px")
      .attr("y", "120px")
      .attr("width", "30px")
      .attr("height", "30px");

    // rear window defrost icon
    tg.append("image")
      .attr("xlink:href", "/images/rear-window-defrost.svg")
      .attr("x", "95px")
      .attr("y", "95px")
      .attr("width", "30px")
      .attr("height", "30px");

    // Initial value setting
    setValue(value, 0); // Set initial value without animation
  };

  const setValue = (value: number, duration: number): void => {
    if (!needleRef.current) return;

    const minAngle = -160;
    const maxAngle = 90;
    const angleRange = maxAngle - minAngle;
    const MAX_GAUGE_VALUE = 8000; // Assuming 8000 RPM as max value (8 × 1000)
    const scaledValue = Math.min(value, MAX_GAUGE_VALUE) / 100; // Convert to 0-80 range

    const angle = minAngle + scale(scaledValue, 80) * angleRange;

    d3.transition()
      .select(() => needleRef.current)
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr("transform", `rotate(${angle})`);

    d3.select("#gauge-value-text").text((value / 1000).toFixed(1));
  };

  return (
    <div
      style={{ width: 500, height: 500 }}
      className="rpm-gauge"
      ref={containerRef}
    ></div>
  );
};

export default RpmGauge;
