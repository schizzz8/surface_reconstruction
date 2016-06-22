#pragma once

#include <iostream>
#include <math.h>
#include <limits>
#include <queue>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <utility>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <functional>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <vtkSmartPointer.h>
#include <vtkVersion.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkXMLImageDataWriter.h>

#include "colors.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class Cell {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Cell(const Eigen::Vector3i& idx=Eigen::Vector3i::Zero()):
        _idx(idx){
        _parent = 0;
        _d = std::numeric_limits<float>::max();
        _u = 1;
    }

    inline bool operator < (const Cell& c) const {
        for (int i=0; i<3; i++){
            if (_idx[i]<c._idx[i])
                return true;
            if (_idx[i]>c._idx[i])
                return false;
        }
        return false;
    }

    inline bool operator == (const Cell& c) const {
        for (int i=0; i<3; i++)
            if(_idx[i] != c._idx[i])
                return false;
        return true;
    }

    inline void setCenter(Eigen::Vector3f origin, float resolution) {
        _center = origin + _idx.cast<float>()*resolution + Eigen::Vector3f(resolution/2,resolution/2,resolution/2);
    }

    Eigen::Vector3i _idx;
    Eigen::Vector3f _center;
    std::vector<size_t> _points;
    Cell* _parent;
    size_t _closest_point;
    float _d;
    Eigen::Vector3f _gradient;
    float _u;
    int _tag;
};

struct QEntry{
    QEntry(Cell* c=0, float d=std::numeric_limits<float>::max()) {
        _cell = c;
        _distance = d;
    }

    inline bool operator < (const QEntry& e) const {
        return e._distance > _distance ;
    }

    float _distance;
    Cell* _cell;
};

struct CellQueue : public std::priority_queue<QEntry> {
    typedef typename std::priority_queue<QEntry>::size_type size_type;
    CellQueue(size_type capacity = 0) { reserve(capacity); }
    inline void reserve(size_type capacity) { this->c.reserve(capacity); }
    inline size_type capacity() const { return this->c.capacity(); }
    inline Cell* top() { return std::priority_queue<QEntry>::top()._cell;}
    inline void push(Cell* c) { return std::priority_queue<QEntry>::push(QEntry(c, c->_d));}
};

class BaseGrid {
public:
    BaseGrid(std::string filename="input.pcd", int prec = 10);
    virtual ~BaseGrid(){}
    inline float resolution(){ return _resolution;}
    inline const Eigen::Vector3i size(){ return _size;}
    inline const Eigen::Vector3f origin(){ return _origin;}
    inline int numCells(){ return _num_cells;}

    virtual void computeDistanceFunction(){}
    virtual void computeGradient(){}
    virtual void computeInitialSurface(){}
    virtual void evolve(size_t num_iter = 0, float deltaT = 0.01){}
    virtual void writeDataToFile(size_t iter = 0){}

protected:
    PointCloud::Ptr _cloud;
    float _resolution;
    float _inverse_resolution;
    Eigen::Vector3f _origin;
    Eigen::Vector3i _size;
    int _num_cells;

    void toIdx(float x, float y, float z, Eigen::Vector3i& idx);
    void toWorld(int i, int j, int k, Eigen::Vector3f& point);
    int toInt(Eigen::Vector3i idx);
    void toIJK(int in, Eigen::Vector3i& idx);
    void toCell(float x, float y, float z, Eigen::Vector3i& idx);
    float solve(std::vector<float> &a);
    inline float euclideanDistance(Eigen::Vector3f a, PointT b){return sqrtf(pow(a.x()-b.x,2)+pow(a.y()-b.y,2)+pow(a.z()-b.z,2));}

    virtual bool hasCell(const Eigen::Vector3i& idx){ return 0;}
    virtual int findNeighbors(Cell** neighbors, Cell* c){ return 0;}
    virtual float interp(float xp, float yp, float zp){return 0;}

private:
    double computeCloudResolution ();
    int getdigit(double number, int digit);
    void manageDirectories(std::string directory);
    int isDirectoryEmpty(const char *dirname);
};

class FullGrid : public BaseGrid {
public:
    FullGrid (std::string filename="input.pcd", int prec = 10);
    ~FullGrid();

    void computeDistanceFunction();
    void computeGradient();
    void computeInitialSurface();
    void evolve(size_t num_iter = 0, float deltaT = 0.01);
    void writeDataToFile(size_t iter=0);

protected:
    Cell* _data;
    Cell*** _cells;

    bool hasCell(const Eigen::Vector3i& idx);
    int findNeighbors(Cell** neighbors, Cell* c);
    float interp(float xp, float yp, float zp);
};

class SparseGrid : public BaseGrid {
public:
    SparseGrid (std::string filename="input.pcd", int prec = 10);

protected:
    std::map<Eigen::Vector3i,Cell*> _cells;

    bool hasCell(const Eigen::Vector3i& idx);
    int findNeighbors(Cell** neighbors, Cell* c);
};

