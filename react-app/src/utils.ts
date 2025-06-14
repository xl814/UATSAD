// Q3 Upper Quartile
export function upperQuartile(arr: number[]): number {
    const sorted = arr.slice().sort((a, b) => a - b);
    const q3Index = Math.floor((sorted.length + 1) * 3 / 4);
    return sorted[q3Index];
}