use charon_audio::{AudioBuffer, AudioKNN, SimdOps};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

fn bench_audio_resampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_resampling");

    for size in [1000, 10000, 100000].iter() {
        let data = Array2::from_shape_fn((2, *size), |(_, i)| i as f32 * 0.001);
        let audio = AudioBuffer::new(data, 44100);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(audio.resample(48000).unwrap()));
        });
    }
    group.finish();
}

fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    for size in [100, 1000, 10000].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i as f32 * 0.5).sin()).collect();

        group.bench_with_input(BenchmarkId::new("multiply", size), size, |bench, _| {
            bench.iter(|| {
                let mut a_copy = a.clone();
                SimdOps::multiply(&mut a_copy, &b);
                black_box(a_copy)
            });
        });

        group.bench_with_input(BenchmarkId::new("rms", size), size, |bench, _| {
            bench.iter(|| black_box(SimdOps::rms(&a)));
        });
    }
    group.finish();
}

fn bench_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_search");

    for n_samples in [100, 1000, 5000].iter() {
        let data = Array2::from_shape_fn((2, *n_samples), |(ch, i)| {
            (i as f32 * 0.01 + ch as f32).sin()
        });

        let query = data.slice(ndarray::s![.., 0..100]);
        let knn = AudioKNN::new(5);

        group.bench_with_input(BenchmarkId::from_parameter(n_samples), n_samples, |b, _| {
            b.iter(|| black_box(knn.find_neighbors(query, data.view())));
        });
    }
    group.finish();
}

fn bench_audio_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_normalization");

    for size in [1000, 10000, 100000].iter() {
        let data = Array2::from_shape_fn((2, *size), |(_, i)| (i as f32 * 0.1).sin() * 0.5);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut audio = AudioBuffer::new(data.clone(), 44100);
                audio.normalize();
                black_box(audio)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_audio_resampling,
    bench_simd_operations,
    bench_knn_search,
    bench_audio_normalization
);
criterion_main!(benches);
