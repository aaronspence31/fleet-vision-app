"use client";
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const DEFAULT_REFRESH_RATE = 1000; // Assuming this value

// Helper functions from the original code
const degToRad = (deg: number): number => {
  return (deg * Math.PI) / 180;
};

const scale = (input: number, max: number): number => {
  return input / max;
};

interface SpeedGaugeProps {
  value: number;
}

const SpeedGauge = ({ value = 0 }: SpeedGaugeProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const needleRef = useRef<SVGPathElement | null>(null);
  const speedTextRef = useRef<SVGTextElement | null>(null);
  const [initialized, setInitialized] = useState(false);

  // Generate the gauge on component mount
  useEffect(() => {
    if (containerRef.current && !initialized) {
      containerRef.current.innerHTML = "";
      generate();
      setInitialized(true);
    }
  }, [containerRef, initialized]);

  // Update values when props change
  useEffect(() => {
    if (needleRef.current && speedTextRef.current) {
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
      { value: 0, visible: false, color: "#FFFFFF" },
      { value: 20, visible: false, color: "#FFFFFF" },
      { value: 30, visible: false, color: "#FD3104" },
      { value: 40, visible: false, color: "#FFFFFF" },
      { value: 50, visible: false, color: "#FD3104" },
      { value: 60, visible: false, color: "#FFFFFF" },
      { value: 80, visible: false, color: "#FFFFFF" },
      { value: 100, visible: false, color: "#FFFFFF" },
      { value: 120, visible: false, color: "#FFFFFF" },
      { value: 140, visible: false, color: "#FFFFFF" },
      { value: 160, visible: false, color: "#FFFFFF" },
      { value: 180, visible: false, color: "#FFFFFF" },
      { value: 200, visible: false, color: "#FFFFFF" },
      { value: 220, visible: false, color: "#FFFFFF" },
      { value: 240, visible: false, color: "#FFFFFF" },
      { value: 260, visible: false, color: "#FFFFFF" },
      { value: 280, visible: false, color: "#FFFFFF" },
      { value: 300, visible: false, color: "#FFFFFF" },
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
    const outerRadius = 200 - 10;
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
    const maxAngle = 150;
    const angleRange = maxAngle - minAngle;

    const ticks = ticksData
      .reduce((acc, curr) => {
        if (curr.value === 0) {
          return [0, 1, 2, 3, 4, 5];
        } else {
          return acc.concat(d3.range(curr.value - 10, curr.value + 10));
        }
      }, [] as number[])
      .filter((d: number) => d % 5 === 0 && d <= 300);

    lg.selectAll("line")
      .data(ticks)
      .enter()
      .append("line")
      .attr("class", "tickline")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", 0)
      .attr("y2", (d: number) => (d % 20 === 0 || d === 0 ? "12" : "7"))
      .attr("transform", (d: number) => {
        const ratio = scale(d, 300);
        const newAngle = minAngle + ratio * angleRange;
        const deviation = d % 20 === 0 || d === 0 ? 12 : 17;
        return `rotate(${newAngle}) translate(0, ${deviation - r})`;
      })
      .style("stroke", (d: number) =>
        d === 30 || d === 50 ? colors[3] : colors[2]
      )
      .style("stroke-width", (d: number) =>
        d % 5 === 0 || d === 0 ? "3" : "1"
      );

    // ticks text
    lg.selectAll("text")
      .data(ticksData)
      .enter()
      .append("text")
      .attr("transform", (d: { value: number; color: string }) => {
        const ratio = scale(d.value, 300);
        const newAngle = degToRad(minAngle + ratio * angleRange);
        const deviation = d.value === 30 || d.value === 50 ? 45 : 50;
        const y = (deviation - r) * Math.cos(newAngle);
        const x = -1 * (deviation - r) * Math.sin(newAngle);
        return `translate(${x}, ${y + 7})`;
      })
      .text((d: { value: number; color: string }) =>
        d.value !== 0 ? d.value : ""
      )
      .attr("fill", (d: { value: number; color: string }) => d.color)
      .attr("font-size", (d: { value: number; color: string }) => {
        return d.value === 30 || d.value === 50 ? "16" : "20";
      })
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

    // speed text in center
    const speedText = tg
      .append("text")
      .text("0")
      .attr("font-size", "80")
      .attr("text-anchor", "middle")
      .attr("fill", colors[2])
      .attr("x", "0")
      .attr("y", "10px")
      .style("position", "absolute")
      .style("z-index", "10");

    speedTextRef.current = speedText.node();

    // km/h text
    tg.append("text")
      .text("km/h")
      .attr("font-size", "16")
      .attr("text-anchor", "middle")
      .attr("fill", colors[2])
      .attr("x", "0")
      .attr("y", "45px")
      .style("position", "absolute")
      .style("z-index", "10");
  };

  const setValue = (value: number, duration: number): void => {
    if (!needleRef.current || !speedTextRef.current) return;

    const minAngle = -160;
    const maxAngle = 150;
    const angleRange = maxAngle - minAngle;
    const angle = minAngle + scale(value, 300) * angleRange;

    d3.select(speedTextRef.current).text(value);

    d3.transition()
      .select(() => needleRef.current)
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr("transform", `rotate(${angle})`);
  };

  return (
    <div
      style={{ width: 500, height: 500 }}
      className="speed-gauge"
      ref={containerRef}
    ></div>
  );
};

export default SpeedGauge;
