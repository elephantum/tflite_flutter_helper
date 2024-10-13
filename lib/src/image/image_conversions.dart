import 'package:image/image.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/src/image/color_space_type.dart';
import 'package:tflite_flutter_helper/src/tensorbuffer/tensorbuffer.dart';

/// Implements some stateless image conversion methods.
///
/// This class is an internal helper.
class ImageConversions {
  static Image convertRgbTensorBufferToImage(TensorBuffer buffer) {
    List<int> shape = buffer.getShape();
    ColorSpaceType rgb = ColorSpaceType.RGB;
    rgb.assertShape(shape);

    int h = rgb.getHeight(shape);
    int w = rgb.getWidth(shape);
    Image image = Image(width: w, height: h);

    List<int> rgbValues = buffer.getIntList();
    assert(rgbValues.length == w * h * 3);

    for (int i = 0, j = 0, wi = 0, hi = 0; j < rgbValues.length; i++) {
      int r = rgbValues[j++];
      int g = rgbValues[j++];
      int b = rgbValues[j++];
      image.setPixelRgba(wi, hi, r, g, b, 1);
      wi++;
      if (wi % w == 0) {
        wi = 0;
        hi++;
      }
    }

    return image;
  }

  static Image convertGrayscaleTensorBufferToImage(TensorBuffer buffer) {
    // Convert buffer into Uint8 as needed.
    TensorBuffer uint8Buffer = buffer.getDataType() == TensorType.uint8
        ? buffer
        : TensorBuffer.createFrom(buffer, TensorType.uint8);

    final shape = uint8Buffer.getShape();
    final grayscale = ColorSpaceType.GRAYSCALE;
    grayscale.assertShape(shape);

    final image = Image.fromBytes(
        width: grayscale.getWidth(shape),
        height: grayscale.getHeight(shape),
        bytes: uint8Buffer.getBuffer().asUint8List().buffer,
        format: Format.uint8);

    return image;
  }

  static void convertImageToTensorBuffer(Image image, TensorBuffer buffer) {
    int w = image.width;
    int h = image.height;
    List<Pixel>? pixelValues =
        image.data?.toList(); // Updated to work with List<Pixel>
    int flatSize = w * h * 3;
    List<int> shape = [h, w, 3];

    if (pixelValues == null) {
      throw ArgumentError('Image data is null.');
    }

    switch (buffer.getDataType()) {
      case TensorType.uint8:
        List<int> byteArr = List.filled(flatSize, 0);
        for (int i = 0, j = 0; i < pixelValues.length; i++) {
          byteArr[j++] = pixelValues[i].r.round(); // Extract red channel
          byteArr[j++] = pixelValues[i].g.round(); // Extract green channel
          byteArr[j++] = pixelValues[i].b.round(); // Extract blue channel
        }
        buffer.loadList(byteArr, shape: shape);
        break;

      case TensorType.float32:
        List<double> floatArr = List.filled(flatSize, 0.0);
        for (int i = 0, j = 0; i < pixelValues.length; i++) {
          floatArr[j++] = pixelValues[i].r.toDouble(); // Extract red channel
          floatArr[j++] = pixelValues[i].g.toDouble(); // Extract green channel
          floatArr[j++] = pixelValues[i].b.toDouble(); // Extract blue channel
        }
        buffer.loadList(floatArr, shape: shape);
        break;

      default:
        throw StateError(
            "${buffer.getDataType()} is unsupported with TensorBuffer.");
    }
  }
}
