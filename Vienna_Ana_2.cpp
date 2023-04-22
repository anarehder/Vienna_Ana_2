#include <array>
#include <deque>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <list>
#include <cmath>
#include <unistd.h>
#include <random>
#include <set>


#include <lsAdvect.hpp>
#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsExpand.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>
#include <lsPrune.hpp>
#include <lsSmartPointer.hpp>
#include <lsToMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsWriteVisualizationMesh.hpp>
#include <lsWriter.hpp>

#include "fourRateInterpolation.hpp"

using namespace std;
constexpr int D = 3;
using NumericType = double;
constexpr NumericType gridDelta = 0.5; //0.2; //0.51; //0.51;//0.27;//0.92;
unsigned outputNum = 0;
using LevelSetType = lsSmartPointer<lsDomain<NumericType, D>>;

constexpr double PI = 3.1415296;

// This function creates a box starting in minCorner spanning to maxCorner
using T = NumericType;
lsSmartPointer<lsMesh<NumericType>> makeBox(hrleCoordType *minCorner, hrleCoordType *maxCorner, double angle) {
  // draw all triangles for the surface and then import from the mesh
  std::vector<std::array<T, 3>> corners;
  corners.resize(std::pow(2, D), {0, 0, 0});

  // first corner is the minCorner
  for (unsigned i = 0; i < D; ++i)
    corners[0][i] = minCorner[i];

  // last corner is maxCorner
  for (unsigned i = 0; i < D; ++i)
    corners.back()[i] = maxCorner[i];

  // calculate all missing corners
  corners[1] = corners[0];
  corners[1][0] = corners.back()[0];

  corners[2] = corners[0];
  corners[2][1] = corners.back()[1];

  if (D == 3) {
    corners[3] = corners.back();
    corners[3][2] = corners[0][2];

    corners[4] = corners[0];
    corners[4][2] = corners.back()[2];

    corners[5] = corners.back();
    corners[5][1] = corners[0][1];

    corners[6] = corners.back();
    corners[6][0] = corners[0][0];
  }

  // now add all corners to mesh
  auto mesh = lsSmartPointer<lsMesh<T>>::New();
  for (unsigned i = 0; i < corners.size(); ++i) {
    mesh->insertNextNode(corners[i]);
  }

  if (D == 2) {
    std::array<unsigned, 2> lines[4] = {{0, 2}, {2, 3}, {3, 1}, {1, 0}};
    for (unsigned i = 0; i < 4; ++i)
      mesh->insertNextLine(lines[i]);
  } else {
    std::array<unsigned, 3> triangles[12] = {
        {0, 3, 1}, {0, 2, 3}, {0, 1, 5}, {0, 5, 4}, {0, 4, 2}, {4, 6, 2},
        {7, 6, 4}, {7, 4, 5}, {7, 2, 6}, {7, 3, 2}, {1, 3, 5}, {3, 7, 5}};
    for (unsigned i = 0; i < 12; ++i)
      mesh->insertNextTriangle(triangles[i]);
  }

  std::array<double, D> axis;
  axis[D-1] = 1.;
  lsTransformMesh(mesh, lsTransformEnum::ROTATION, axis, angle).apply();

  return mesh;
}

void writeSurface(LevelSetType LS, std::string fileName) {
  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToSurfaceMesh<NumericType, D>(LS, mesh).apply();
  lsVTKWriter<NumericType>(mesh, fileName).apply();
}

void writeLS(LevelSetType LS, std::string fileName) {
  auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  lsToMesh<NumericType, D>(LS, mesh).apply();
  lsVTKWriter<NumericType>(mesh, fileName).apply();
}

class WetEtch : public lsVelocityField<NumericType> {
      static constexpr std::array<NumericType, D> direction100{0.707106781187,0.707106781187,0};
      // static constexpr std::array<NumericType, D> direction100{1,0.,0};
      static constexpr std::array<NumericType, D> direction010{-0.707106781187,0.707106781187,0};

      NumericType r100 = 0.013283; // 0.0166666666667;
      NumericType r110 = 0.024167; // 0.0309166666667;
      NumericType r111 = 0.000083; // 0.000121666666667;
      NumericType r311 = 0.023933; // 0.0300166666667;

      std::vector<double> velocities;

  public:
  WetEtch(std::vector<double> vel) : velocities(vel) {}

  double getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
                           int material,
                           const std::array<NumericType, 3> &normal,
                           unsigned long /* pointID */) final {
    if(std::abs(velocities[material]) < 1e-3) {
      return 0;
    }

    return velocities[material] * fourRateInterpolation<double,3>(normal, direction100, direction010, r100, r110, r111, r311);
  }
};

double corrosaoAnisotropica(std::set<int> positions){

    omp_set_num_threads(24);

    LevelSetType substrate;
    hrleCoordType extent = 20;
    int dimensao = 6;

    {
      hrleCoordType bounds[2 * D] = {-extent, extent, -extent, extent};
      lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
      boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
      boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
      boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
      substrate = LevelSetType::New(bounds, boundaryCons, gridDelta);
      NumericType origin[D] = {};
      origin[D-1] = 0.01;
      NumericType planeNormal[D] = {};
      planeNormal[D-1] = 1.;
      lsMakeGeometry(substrate, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal)).apply();
    }

  // make mask
    auto mask = LevelSetType::New(substrate->getGrid());
    double maskAngle = 0.;
    {
        NumericType origin[D] = {};
        origin[D-1] = 2.;
        NumericType planeNormal[D] = {};
        planeNormal[D-1] = 1.;
        auto plane = LevelSetType::New(mask->getGrid());
        lsMakeGeometry(plane, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal)).apply();

        origin[D-1] = 0.;
        planeNormal[D-1] = -1.;
        lsMakeGeometry(mask, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal)).apply();

        lsBooleanOperation(mask, plane, lsBooleanOperationEnum::INTERSECT).apply();

        // FAZENDO UMA MÁSCARA COM O TAMANHO DE S (S.SIZE)
        for (int block : positions){
            auto maskHole = LevelSetType::New(mask->getGrid());
            //ENCONTRAR AS COORDENADAS DO PONTO SORTEADO
            hrleCoordType block_double = static_cast<hrleCoordType>(block);
            hrleCoordType dimensao_double = static_cast<hrleCoordType>(dimensao);
            hrleCoordType calculoX = static_cast<hrleCoordType>(std::ceil(block_double/dimensao_double));
            hrleCoordType calculoY = static_cast<hrleCoordType>(block%dimensao);
            //cout << "Posição " << block << " \n";
            //cout << "calculoX " << calculoX << " \n";
            //cout << "calculoY " << calculoY << " \n";

            hrleCoordType coordX = -extent + (calculoX*5);
            hrleCoordType coordY = -extent + (calculoY*5);
            if(calculoY == 0){
                coordY = (extent/2);
            }

            //cout << "X " << coordX << " \n";
            //cout << "Y " << coordY << " \n";

            hrleCoordType minCorner[D] = {coordX, coordY, -1.};
            hrleCoordType maxCorner[D] = {coordX+5, coordY+5, 3.};

            auto mesh = makeBox(minCorner, maxCorner, maskAngle);
            lsFromSurfaceMesh(maskHole, mesh).apply();
            lsBooleanOperation(mask, maskHole, lsBooleanOperationEnum::RELATIVE_COMPLEMENT).apply();
            writeSurface(mask, "MASK.vtp");
        }

    }

    lsBooleanOperation(substrate, mask, lsBooleanOperationEnum::UNION).apply();

    std::vector<double> epiVels = {0, 15};
    auto velocity = lsSmartPointer<WetEtch>::New(epiVels);

    std::vector<LevelSetType> LSs;
    LSs.push_back(mask);
    LSs.push_back(substrate);

    lsAdvect<NumericType, D> advection(LSs, velocity);
    advection.setSingleStep(true);
    advection.setSaveAdvectionVelocities(true);
    advection.setIntegrationScheme(lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);
    double time = 15; //51.; //41;
    double nextOutput = 40.5;
    unsigned counter = 1;
    while(time > 0.01) {
      advection.setAdvectionTime(time);
      time -= advection.getAdvectedTime();
      if(time < nextOutput) {
        nextOutput -= 10.;
        ++counter;
      }
      advection.apply();
    }

    NumericType origin[D] = {};
    origin[D-1] = -1.;
    NumericType planeNormal[D] = {};
    planeNormal[D-1] = 1.;
    auto plane1 = LevelSetType::New(mask->getGrid());
    lsMakeGeometry(plane1, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal)).apply();

    lsBooleanOperation(substrate, plane1, lsBooleanOperationEnum::INTERSECT).apply();
    writeSurface(substrate, "ARQ_FINAL.vtp");
    //writeSurface(substrate, "TESTE_" + std::to_string(i) + ".vtp");

    lsWriteVisualizationMesh<NumericType, D> visMesh;
    visMesh.insertNextLevelSet(mask);
    visMesh.insertNextLevelSet(substrate);
    if(maskAngle < 0.01) {
      visMesh.setFileName("aligned");
    } else {
      visMesh.setFileName("misaligned");
    }
    visMesh.apply();


    int res0;
// ALTERANDO O NOME DO ARQUIVO GERADO
    res0 = system("mv aligned_volume.vtu DISP_TESTE.vtu");

//CRIAR UMA COPIA DO ARQUIVO cp arquivo novo_arquivo ->cp aligned_volume.vtu DISP_TESTE.vtu


// ALTERANDO PARA STL
    int res1;
    res1 = system("jupyter nbconvert --to notebook --execute makeSTL.ipynb");

// FAZENDO A COMPARAÇÃO
    int res2;
    res2 = system("openscad -o DISP_SUBTRAIDO.stl DIFFERENCE.scad");

// CALCULANDO O VOLUME
    int res3;
    res3 = system("jupyter nbconvert --to notebook --execute volume.ipynb");

// OBTENDO O VOLUME CALCULADO
 // FAZER O CALCULO IGUAL O LER ARQ
    char texto[100];
    char *result;
    int ip;

//para declarar um arquivo
    FILE *fp;

// MODELO - fp = fopen(const char filename,const char mode);
    fp = fopen("VOLUME.txt","r");

    if (fp == NULL)  // Se houve erro na abertura
      {
        printf("Problemas na abertura do arquivo\n");
      }

    while (!feof(fp)) /* Enquanto não se chegar no final do arquivo */
      {
	  // Lê uma linha (inclusive com o '\n')
        result = fgets(texto, 100, fp);  // o 'fgets' lê até 99 caracteres ou até o '\n'
      }

    double VOLUME = atof(texto);
    //double vol_dif = std::stod(VOLUME);
    printf("Volume é %f\n", VOLUME );
    fclose(fp);



    /* if (VOLUME < 1){
      std::cout << " Estrutura final encontrada, diferença de volume:" << VOLUME << endl;
      break;
    } */
    //return vol_dif;
    return VOLUME;
}

int main() {

    // Obter o tempo atual
    auto start = std::chrono::high_resolution_clock::now();
    int i = 1; // contagem de iterações
    std::set<int> allPositions;

    for (int j = 1; j <= 36; j++) {
        allPositions.insert(j);
    }
    //std::set<int> positions = {1, 2, 3, 4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 28, 31, 32, 33, 34, 35, 36};
    //std::set<int> positions = {7, 9, 10, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 29, 30};
    //std::set<int> positions = {2, 8, 9, 10, 11, 17, 20, 21, 22, 23,26, 32, 33, 34, 35, 36};
    //std::set<int> positionsToExclude = {3,9,14,15,16,20,21,22,26,27,28};


    double difference = 100;
    // quando for um ponto position tem apenas 1 valor, mas preciso alterar pra quando for um array ou um set de pontos.
    while (difference > 2){
        std::cout << "Numero de iterações: " << i << std::endl;
        std::set<int> positionsToExclude;
        std::set<int> positions;
        //std::set<int> positions = {1,2,4,5,6,7,8,10,11,12,13,17,18,19,23,24,25,29,30,31,32,33,34,35,36};
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(1, 36);

        while (positionsToExclude.size() < 11) {
            int position = distr(gen);
            positionsToExclude.insert(position);
        }
        for (int valor : positionsToExclude){
            std::cout << valor << " ";
        }
        std::cout << std::endl;
        for (const auto& numero : allPositions) {
                if (positionsToExclude.count(numero) == 0) {
                    positions.insert(numero);
                }
        }
        difference = corrosaoAnisotropica(positions);
        i++;
    }

    cout << "Estrutura final encontrada, com diferença de volume = " << difference << endl;

    // Obter o tempo final e calcular a diferença
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double tempo = duration.count();
    double seconds = tempo / 1000000;

    std::cout << "Tempo de código total: " << seconds << " segundos" << std::endl;
    // Imprimir o tempo de execução
    //std::cout << "Tempo de execucao: " << duration.count() << " microsegundos" << std::endl;

    return 0;
}
